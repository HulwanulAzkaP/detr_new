CONFIG = {
    "model": {
        "backbone": "custom_detr",  # Nama backbone custom
        "pretrained": False         # Tidak menggunakan pretrained weights
    },
    "training": {
        "epochs": 20,               # Jumlah epoch
        "batch_size": 32,           # Ukuran batch
        "learning_rate": 0.001,     # Learning rate
        "optimizer": "adam",        # Optimizer yang digunakan
        "device": "cuda"            # Gunakan 'cuda' jika GPU tersedia
    },
    "dataset": {
        "train": {
            "images": "./dataset/train",
            "annotations": "./dataset/train/_annotations.coco.json"
        },
        "valid": {
            "images": "./dataset/valid",
            "annotations": "./dataset/valid/_annotations.coco.json"
        },
        "test": {
            "images": "./dataset/test",
            "annotations": "./dataset/test/_annotations.coco.json"
        },
        "num_classes": 2            # Dua kelas: api dan asap
    },
    "output": {
        "checkpoint_path": "./checkpoints",  # Folder untuk menyimpan model
        "log_path": "./logs"                # Folder untuk menyimpan log
    }
}
