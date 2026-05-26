"""Coarse label mapping for ImageNet-1K classification outputs.

The pretrained models in this project output ImageNet-1K labels.  The uploaded
course dataset uses coarse folder labels such as cat/dog/car/ship.  To compute a
meaningful accuracy, we map fine-grained ImageNet predictions back to these
coarse dataset labels.

You can edit COARSE_KEYWORDS / ID_RANGES if your dataset has more classes.
"""

from __future__ import annotations

import re
from typing import Iterable


def normalize_label(text: str) -> str:
    text = str(text).lower().strip()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text


# ImageNet class id ranges in the usual torchvision category order.
# These ranges are intentionally conservative and only used for very clear groups.
DOG_ID_RANGE = range(151, 269)       # Chihuahua ... Mexican hairless
CAT_ID_RANGE = range(281, 286)       # tabby ... Egyptian cat
BIRD_ID_RANGES = [range(7, 25), range(80, 101)]


COARSE_KEYWORDS = {
    "airplane": [
        "airliner", "warplane", "airship", "aircraft", "plane", "jet",
        "wing", "missile", "projectile"
    ],
    "bird": [
        "bird", "cock", "hen", "ostrich", "brambling", "goldfinch", "house finch",
        "junco", "indigo bunting", "robin", "bulbul", "jay", "magpie", "chickadee",
        "water ouzel", "kite", "bald eagle", "vulture", "great grey owl", "black grouse",
        "ptarmigan", "ruffed grouse", "prairie chicken", "peacock", "quail", "partridge",
        "african grey", "macaw", "sulphur crested cockatoo", "lorikeet", "coucal",
        "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "drake",
        "red breasted merganser", "goose", "black swan", "flamingo", "little blue heron",
        "american egret", "bittern", "crane", "limpkin", "european gallinule",
        "american coot", "bustard", "ruddy turnstone", "red backed sandpiper",
        "redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross"
    ],
    "car": [
        "car", "sports car", "racer", "cab", "taxi", "limousine", "convertible",
        "jeep", "minivan", "beach wagon", "station wagon", "pickup", "police van",
        "moving van", "ambulance", "model t", "go kart", "go-kart", "landrover",
        "tow truck", "trailer truck", "fire engine"
    ],
    "cat": [
        "cat", "tabby", "tiger cat", "persian cat", "siamese cat", "egyptian cat", "lynx"
    ],
    "dog": [
        "dog", "puppy", "hound", "terrier", "retriever", "shepherd", "spaniel", "collie",
        "chihuahua", "mastiff", "poodle", "husky", "malamute", "samoyed", "beagle",
        "dalmatian", "schnauzer", "setter", "boxer", "rottweiler", "corgi", "pug"
    ],
    "flower": [
        "flower", "daisy", "yellow lady", "lady's slipper", "orchid", "rose", "sunflower",
        "tulip", "poppy", "lotus", "lily", "rapeseed", "pot", "vase"
    ],
    "ship": [
        "ship", "liner", "container ship", "aircraft carrier", "pirate", "submarine",
        "boat", "speedboat", "lifeboat", "canoe", "trimaran", "catamaran", "gondola",
        "yawl", "schooner", "dock", "breakwater"
    ],
}


ALIASES = {
    "airplane": "airplane",
    "aeroplane": "airplane",
    "plane": "airplane",
    "aircraft": "airplane",
    "automobile": "car",
    "auto": "car",
    "vehicle": "car",
    "kitty": "cat",
    "kitten": "cat",
    "puppy": "dog",
    "boat": "ship",
    "vessel": "ship",
    "flowers": "flower",
}


def canonical_dataset_label(label: str) -> str:
    label = normalize_label(label)
    return ALIASES.get(label, label)


def imagenet_to_coarse(class_id: int, imagenet_label: str, dataset_labels: Iterable[str] | None = None) -> str:
    """Map one ImageNet prediction to a coarse dataset label.

    If dataset_labels is given, only those labels are returned; otherwise common coarse
    labels from COARSE_KEYWORDS may be returned.  If no mapping is found, returns
    "other".
    """
    label = normalize_label(imagenet_label)

    # ID-based mappings are robust against spelling variants.
    if int(class_id) in DOG_ID_RANGE:
        candidate = "dog"
    elif int(class_id) in CAT_ID_RANGE:
        candidate = "cat"
    elif any(int(class_id) in r for r in BIRD_ID_RANGES):
        candidate = "bird"
    else:
        candidate = "other"

    if candidate == "other":
        for coarse, keywords in COARSE_KEYWORDS.items():
            if any(k in label for k in keywords):
                candidate = coarse
                break

    if dataset_labels is not None:
        allowed = {canonical_dataset_label(x) for x in dataset_labels}
        if candidate in allowed:
            return candidate
        # Fallback: exact/substring matching for custom folder names.
        for coarse in allowed:
            if coarse != "other" and (coarse in label or label in coarse):
                return coarse
        return "other"
    return candidate


def topk_contains_label(results: list[dict], true_label: str, dataset_labels: Iterable[str] | None = None) -> bool:
    target = canonical_dataset_label(true_label)
    for item in results:
        pred = imagenet_to_coarse(item["class_id"], item["label"], dataset_labels)
        if pred == target:
            return True
    return False
