"""
Step 5 - image search with CLIP embeddings (multimodal RAG, retrieval part).

CLIP maps images AND texts into the same vector space. So pillar 2 works for images unchanged:
embed every image, store the vectors in Chroma, then search with a text query or with another
image. Typical industrial uses: "find all past images that look like this defect", building a
dataset to label, deduplicating, finding similar incidents in a photo archive.

Usage: python 05_image_search.py   (no API call, first run downloads CLIP, ~600MB)
"""
import chromadb
from PIL import Image
from ultralytics.utils import ASSETS

from cv_tools import clip_model
from vision_utils import PLATES_DIR, ROOT, load_labels

labels = load_labels()
model = clip_model()

# --- index ----------------------------------------------------------------------
paths = sorted(PLATES_DIR.glob("plate_*.jpg")) + [ASSETS / "bus.jpg", ASSETS / "zidane.jpg"]
client = chromadb.PersistentClient(path=str(ROOT / "image_db"))
collection = client.get_or_create_collection("images", metadata={"hnsw:space": "cosine"})

if collection.count() != len(paths):
    print(f"Embedding {len(paths)} images with CLIP ...")
    embeddings = model.encode([Image.open(p).convert("RGB") for p in paths], normalize_embeddings=True,
                              batch_size=16, show_progress_bar=True)
    collection.upsert(
        ids=[p.name for p in paths],
        embeddings=embeddings.tolist(),  # we give the vectors ourselves, no embedding function needed
        metadatas=[{"path": str(p), "defect": labels.get(p.name, {}).get("defect", "not a plate")} for p in paths],
    )
print(f"{collection.count()} images indexed\n")


def search_text(query, k=5):
    vec = model.encode([query], normalize_embeddings=True).tolist()
    return collection.query(query_embeddings=vec, n_results=k)


def search_image(path, k=6):
    vec = model.encode([Image.open(path).convert("RGB")], normalize_embeddings=True).tolist()
    return collection.query(query_embeddings=vec, n_results=k)


def show(res):
    for img_id, dist, meta in zip(res["ids"][0], res["distances"][0], res["metadatas"][0]):
        print(f"    {dist:.3f}  {img_id:<16} ({meta['defect']})")


# --- text -> image ----------------------------------------------------------------
for query in ["a red double-decker bus in a street", "a man in a suit and tie",
              "a metal plate with a brown stain", "a metal plate with a missing screw",
              "a blurry photo", "a very dark photo"]:
    print(f"Text: {query!r}")
    show(search_text(query))
    print()

# --- image -> image -------------------------------------------------------------
for defect in ["stain", "missing_screw", "scratch"]:
    example = next(f for f, x in sorted(labels.items()) if x["defect"] == defect and not x["imaging_issue"]
                   and not x["subtle"])
    res = search_image(PLATES_DIR / example)
    same = sum(m["defect"] == defect for m in res["metadatas"][0][1:])
    print(f"Image: {example} ({defect}) -> {same}/{len(res['ids'][0]) - 1} of the nearest neighbours are also {defect}")
    show(res)
    print()

# What you should see:
#  - the bus and the man are found instantly: CLIP was trained on web photos + captions
#  - "blurry" / "dark" work well too: global properties of the image
#  - defect queries are much weaker: all plates look alike globally, a 3 cm stain barely moves
#    the embedding. Image -> image is usually better than text -> image for that.
#  - to make this work on defects: embed CROPS around anomalies instead of full images
#    (try it with cv_tools.find_anomalies + crop_region), or fine-tune the embedding model.
#
# The full multimodal RAG: retrieve similar past images + their notes/reports, and give them
# to the VLM as context ("here are 2 past cases that look similar and how they were handled").
