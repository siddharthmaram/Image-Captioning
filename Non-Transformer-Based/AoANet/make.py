import json

# Load your original JSON file
with open('/home/sri/AoANet_final/AoANet/data/dataset_custom_combined.json', 'r') as f:
    data = json.load(f)

coco_format = {
    "info": {},
    "licenses": [],
    "images": [],
    "annotations": []
}

annotation_id = 0

for image in data["images"]:
    # Add image info
    coco_format["images"].append({
        "id": image["imgid"],
        "file_name": image["filename"]
    })

    # Add each caption annotation
    for sent in image["sentences"]:
        annotation_id += 1
        coco_format["annotations"].append({
            "id": annotation_id,
            "image_id": image["imgid"],
            "caption": sent["raw"]
        })

# Save converted JSON
with open('bio_coco_format.json', 'w') as f:
    json.dump(coco_format, f)
