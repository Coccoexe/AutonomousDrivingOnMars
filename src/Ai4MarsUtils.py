import numpy as np

COLOR_NAMES = {0: 'soil', 1: 'bedrock', 2: 'sand', 3: 'big rocks', 255: 'no label'}
COLOR_MAP = {0: [0, 0, 0], 1: [128, 128, 128], 2: [0, 165, 255], 3: [0, 0, 255], 255: [255, 255, 255]}

def err(message: str) -> None:
    print('\033[91m' + message + '\033[0m')
    return

def gray2color_s(img: np.array) -> np.array:
        colored = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for k, v in COLOR_MAP.items():
            colored[img == k] = v
        return colored
    
def gray2color(imgs: list[np.array]) -> list[np.array]:
        return [gray2color_s(img) for img in imgs]