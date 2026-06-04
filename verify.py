import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

model = SentenceTransformer(
    "sentence-transformers/stsb-bert-large"
)

def l2_normalize(x):

    return x / np.linalg.norm(
        x,
        axis=1,
        keepdims=True
    )

star_file = Path("STAR/text/ntu120_part_descriptions.txt")

with open(star_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

head_texts = []
hand_texts = []
arm_texts = []
hip_texts = []
leg_texts = []
foot_texts = []

class_names = []

for line in lines[:60]:
    line = line.strip()
    parts = line.split(";")
    class_name = parts[0].strip()
    class_names.append(class_name)
    body_parts = {}

    for p in parts[1:]:
        if ":" in p:
            key, value = p.split(":", 1)
            key = key.strip()
            value = value.strip()
            body_parts[key] = value

    head_texts.append(body_parts.get("Head", ""))
    hand_texts.append(body_parts.get("Hand", ""))
    arm_texts.append(body_parts.get("Arm", ""))
    hip_texts.append(body_parts.get("Hip", ""))
    leg_texts.append(body_parts.get("Leg", ""))
    foot_texts.append(body_parts.get("Foot", ""))

print("\nFIRST 5 CLASSES:\n")

for i in range(5):
    print(f"{i} -> {class_names[i]}")

print("\nHEAD EXAMPLE:\n")
print(head_texts[0])

print("\nHAND EXAMPLE:\n")
print(hand_texts[0])

print("\nTOTAL CLASSES:")
print(len(class_names))

print("\nGenerating embeddings...\n")

head_embeddings = model.encode(head_texts)
hand_embeddings = model.encode(hand_texts)
arm_embeddings = model.encode(arm_texts)
hip_embeddings = model.encode(hip_texts)
leg_embeddings = model.encode(leg_texts)
foot_embeddings = model.encode(foot_texts)

print("Embeddings generated successfully!")

print("\nEmbedding Shapes:\n")

print("Head:", head_embeddings.shape)
print("Hand:", hand_embeddings.shape)
print("Arm:", arm_embeddings.shape)
print("Hip:", hip_embeddings.shape)
print("Leg:", leg_embeddings.shape)
print("Foot:", foot_embeddings.shape)

head_embeddings = l2_normalize(head_embeddings)
hand_embeddings = l2_normalize(hand_embeddings)
arm_embeddings = l2_normalize(arm_embeddings)
hip_embeddings = l2_normalize(hip_embeddings)
leg_embeddings = l2_normalize(leg_embeddings)
foot_embeddings = l2_normalize(foot_embeddings)

print("\nNorm Check:\n")

print(np.linalg.norm(head_embeddings[0]))
print(np.linalg.norm(hand_embeddings[0]))

np.save("ntu60_parts/head_embeddings.npy", head_embeddings)
np.save("ntu60_parts/hand_embeddings.npy", hand_embeddings)
np.save("ntu60_parts/arm_embeddings.npy", arm_embeddings)
np.save("ntu60_parts/hip_embeddings.npy", hip_embeddings)
np.save("ntu60_parts/leg_embeddings.npy", leg_embeddings)
np.save("ntu60_parts/foot_embeddings.npy", foot_embeddings)

print("\nAll embeddings saved successfully!")

loaded = np.load("ntu60_parts/head_embeddings.npy")

print("\nLoaded Shape:")
print(loaded.shape)