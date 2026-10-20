"""
Fase 3: red neuronal propia para los guardrails, diseñada y entrenada desde cero (sin preentrenamiento).

Arquitectura:
    frase → tokens (palabras) → Embedding(128) → BiLSTM(2 x 128) → pooling media + máximo
          → Linear(512 → 128) → ReLU → Dropout → Linear(128 → 3 clases)

Sirve de punto de comparación con MrBERT-biomed: mismo dataset, mismo split por analito y misma
evaluación. La diferencia mide cuánto aporta el preentrenamiento frente a una red que solo
aprende de nuestras ~900 frases.

Uso:
    python training/guardrails/red_propia.py [--epocas 20] [--lr 2e-3]
"""
import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

from dataset import CLASES, cargar_dataset, parsear_pesos, resumen  # noqa: E402

SALIDA = RAIZ / "models" / "guardrails" / "bilstm"
PAD, UNK = 0, 1
MAX_TOKENS = 60


def tokenizar(texto):
    """Palabras en minúscula; se conservan las tildes ('tú' y 'tu' significan distinto)."""
    return re.findall(r"\w+", texto.lower())


def construir_vocabulario(frases, min_frecuencia=2):
    conteo = Counter(t for f in frases for t in tokenizar(f))
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for token, n in sorted(conteo.items(), key=lambda x: (-x[1], x[0])):
        if n >= min_frecuencia:
            vocab[token] = len(vocab)
    return vocab


def codificar(frases, vocab):
    """Devuelve un tensor (n, largo) relleno con <pad>, truncado a MAX_TOKENS."""
    ids = [[vocab.get(t, UNK) for t in tokenizar(f)][:MAX_TOKENS] or [UNK] for f in frases]
    largo = max(map(len, ids))
    return torch.tensor([x + [PAD] * (largo - len(x)) for x in ids])


class ClasificadorBiLSTM(nn.Module):
    def __init__(self, tam_vocab, dim_embedding=128, dim_oculta=128, num_clases=3, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(tam_vocab, dim_embedding, padding_idx=PAD)
        self.lstm = nn.LSTM(dim_embedding, dim_oculta, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.clasificador = nn.Sequential(
            nn.Linear(4 * dim_oculta, dim_oculta), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim_oculta, num_clases))

    def forward(self, ids):
        mascara = (ids != PAD).unsqueeze(-1)                     # (n, largo, 1)
        largos = mascara.sum(1).squeeze(-1).cpu()
        # Empaquetar evita que la dirección hacia atrás de la LSTM lea primero el relleno
        empaquetado = nn.utils.rnn.pack_padded_sequence(
            self.dropout(self.embedding(ids)), largos, batch_first=True, enforce_sorted=False)
        salida, _ = self.lstm(empaquetado)
        salida, _ = nn.utils.rnn.pad_packed_sequence(salida, batch_first=True, total_length=ids.shape[1])  # (n, largo, 2 * oculta)
        # Media y máximo sobre los tokens reales, ignorando el relleno
        media = (salida * mascara).sum(1) / mascara.sum(1).clamp(min=1)
        maximo = salida.masked_fill(~mascara, -1e4).max(1).values
        return self.clasificador(torch.cat([media, maximo], dim=-1))


def probabilidades(frases, ruta=SALIDA):
    """Probabilidad de cada clase por frase (softmax de la salida), en el orden de CLASES."""
    vocab = json.loads((ruta / "vocab.json").read_text(encoding="utf-8"))
    config = json.loads((ruta / "config.json").read_text(encoding="utf-8"))
    modelo = ClasificadorBiLSTM(**config)
    modelo.load_state_dict(torch.load(ruta / "modelo.pt", weights_only=True))
    modelo.eval()
    with torch.no_grad():
        return [p for i in range(0, len(frases), 128)
                for p in modelo(codificar(frases[i:i + 128], vocab)).softmax(-1).tolist()]


def predecir(frases, ruta=SALIDA):
    return [max(range(len(p)), key=p.__getitem__) for p in probabilidades(frases, ruta)]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--epocas", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--salida", type=Path, default=SALIDA)
    parser.add_argument("--pesos", default="1,1,1",
                        help="Pesos de la entropía cruzada por clase (informativa,diagnostica,alarmista), ej. 1,2,2")
    parser.add_argument("--paciencia", type=int, default=5, help="Épocas sin mejorar en validación antes de parar")
    args = parser.parse_args()

    from sklearn.metrics import f1_score

    torch.manual_seed(args.seed)
    dataset, motivos = cargar_dataset()
    print(resumen(dataset, motivos), flush=True)
    # La red propia se compara con el v1: solo frases sintéticas
    train = [d for d in dataset if d["split"] == "train" and d["fuente"] == "sintetica"]
    val = [d for d in dataset if d["split"] == "val" and d["fuente"] == "sintetica"]

    vocab = construir_vocabulario([d["frase"] for d in train])
    config = {"tam_vocab": len(vocab), "dim_embedding": 128, "dim_oculta": 128, "num_clases": len(CLASES), "dropout": 0.3}
    modelo = ClasificadorBiLSTM(**config)
    parametros = sum(p.numel() for p in modelo.parameters())
    print(f"Vocabulario: {len(vocab)} palabras; parámetros: {parametros:,}", flush=True)

    x_val = codificar([d["frase"] for d in val], vocab)
    y_val = [d["etiqueta"] for d in val]
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=0.01)
    pesos = parsear_pesos(args.pesos)
    perdida_fn = nn.CrossEntropyLoss(weight=torch.tensor(pesos))

    mejor_f1, mejor_estado, sin_mejorar, historial = -1, None, 0, []
    inicio = time.time()
    for epoca in range(1, args.epocas + 1):
        modelo.train()
        orden = torch.randperm(len(train)).tolist()
        total = 0.0
        for i in range(0, len(orden), args.batch):
            lote = [train[j] for j in orden[i:i + args.batch]]
            x = codificar([d["frase"] for d in lote], vocab)
            y = torch.tensor([d["etiqueta"] for d in lote])
            optimizador.zero_grad()
            perdida = perdida_fn(modelo(x), y)
            perdida.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            optimizador.step()
            total += perdida.item() * len(lote)

        modelo.eval()
        with torch.no_grad():
            f1 = f1_score(y_val, modelo(x_val).argmax(-1).tolist(), average="macro")
        historial.append({"epoca": epoca, "perdida_train": round(total / len(train), 4), "f1_val": round(f1, 4)})
        print(f"  época {epoca:>2}: pérdida {total / len(train):.4f}  F1 val {f1:.3f}", flush=True)
        if f1 > mejor_f1:
            mejor_f1, sin_mejorar = f1, 0
            mejor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejorar += 1
            if sin_mejorar >= args.paciencia:
                print(f"  sin mejora en {args.paciencia} épocas: se detiene", flush=True)
                break
    minutos = (time.time() - inicio) / 60

    salida = args.salida.resolve()
    salida.mkdir(parents=True, exist_ok=True)
    torch.save(mejor_estado, salida / "modelo.pt")
    (salida / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    (salida / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with open(salida / "entrenamiento.json", "w", encoding="utf-8") as f:
        json.dump({"arquitectura": "Embedding + BiLSTM + pooling media/máximo + MLP", "parametros": parametros,
                   "frases_train": len(train), "frases_val": len(val), "seed": args.seed, "pesos_clase": pesos, "lr": args.lr, "batch": args.batch,
                   "mejor_f1_val": mejor_f1, "minutos": round(minutos, 2), "historial": historial},
                  f, indent=2, ensure_ascii=False)
    print(f"Mejor F1 macro en validación: {mejor_f1:.3f}")
    print(f"Modelo guardado en {salida.relative_to(RAIZ)} ({minutos:.1f} min)")


if __name__ == "__main__":
    main()
