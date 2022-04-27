import gdown

def load_bb():
    url = "https://drive.google.com/uc?id=1BX2wrhKHDovILBhEYA_UQXxwZc7P4jkH"
    output = "dino-bb.zip"
    gdown.download(url,output,quiet=False)

def load_finetuned_model_dino():
    url = ""
    return

load_bb()
