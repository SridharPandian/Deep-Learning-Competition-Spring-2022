from torchvision import transforms as T

def basic_transform_function(image_size, mean, std):
    return T.Compose([
        T.ToTensor(),
        T.Resize((image_size, image_size)),
        T.Normalize(
            mean = mean,
            std = std
        )
    ])