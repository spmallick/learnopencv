import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import gradio as gr
import pickle


class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        # self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.itos = {0: "pad", 1: "startofseq", 2: "endofseq", 3: "unk"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.index = 4

    def __len__(self):
        return len(self.itos)

    def tokenizer(self, text):
        text = text.lower()
        tokens = re.findall(r"\w+", text)
        return tokens

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        for sentence in sentence_list:
            tokens = self.tokenizer(sentence)
            frequencies.update(tokens)

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = self.index
                self.itos[self.index] = word
                self.index += 1

    def numericalize(self, text):
        tokens = self.tokenizer(text)
        numericalized = []
        for token in tokens:
            if token in self.stoi:
                numericalized.append(self.stoi[token])
            else:
                numericalized.append(self.stoi["<unk>"])
        return numericalized


# You'll need to ensure these match your train.py
EMBED_DIM = 256
HIDDEN_DIM = 512
MAX_SEQ_LENGTH = 25
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"

# Where you saved your model in train.py
# MODEL_SAVE_PATH = "best_checkpoint.pth"
MODEL_SAVE_PATH = "final_model.pth"

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

print(vocab)

vocab_size = len(vocab)

print(vocab_size)


# -----------------------------------------------------------------
# 2. Model (Must match structure in train.py)
# -----------------------------------------------------------------
class ResNetEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = True
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.fc = nn.Linear(resnet.fc.in_features, embed_dim)
        self.batch_norm = nn.BatchNorm1d(embed_dim, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)  # (batch_size, 2048, 1, 1)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        features = self.batch_norm(features)
        return features


class DecoderLSTM(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions, states):
        # remove the last token for input
        captions_in = captions
        emb = self.embedding(captions_in)
        features = features.unsqueeze(1)

        # print(features.shape)
        # print(emb.shape)

        lstm_input = torch.cat((features, emb), dim=1)
        outputs, returned_states = self.lstm(lstm_input, states)
        logits = self.fc(outputs)
        return logits, returned_states

    def generate(self, features, max_len=20):
        """
        Greedy generation from the features as initial context.
        """
        batch_size = features.size(0)
        states = None
        generated_captions = []

        start_idx = 1  # <start>
        end_idx = 2  # <end>

        inputs = features
        # current_tokens = torch.LongTensor([start_idx] * batch_size).to(features.device).unsqueeze(0)
        current_tokens = [start_idx]

        for _ in range(max_len):
            input_tokens = torch.LongTensor(current_tokens).to(features.device).unsqueeze(0)
            logits, states = self.forward(inputs, input_tokens, states)

            logits = logits.contiguous().view(-1, vocab_size)
            predicted = logits.argmax(dim=1)[-1].item()

            generated_captions.append(predicted)
            current_tokens.append(predicted)

            # check if all ended
            # all_ended = True
            # for i, w in enumerate(predicted.numpy()):
            #     print(w)
            #     if w != end_idx:
            #         all_ended = False
            #         break
            # if all_ended:
            #     break

        return generated_captions


class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def generate(self, images, max_len=MAX_SEQ_LENGTH):
        features = self.encoder(images)
        return self.decoder.generate(features, max_len=max_len)


# -----------------------------------------------------------------
# 3. LOAD THE TRAINED MODEL
# -----------------------------------------------------------------
def load_trained_model():
    encoder = ResNetEncoder(embed_dim=EMBED_DIM)
    decoder = DecoderLSTM(EMBED_DIM, HIDDEN_DIM, vocab_size)
    model = ImageCaptioningModel(encoder, decoder).to(DEVICE)

    # Load weights from disk
    state_dict = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict["model_state_dict"])
    model.eval()
    # print(model)
    return model


model = load_trained_model()

# -----------------------------------------------------------------
# 4. INFERENCE FUNCTION (FOR GRADIO)
# -----------------------------------------------------------------
transform_inference = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def generate_caption_for_image(img):
    """
    Gradio callback: takes a PIL image, returns a string caption.
    """
    pil_img = img.convert("RGB")
    img_tensor = transform_inference(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output_indices = model.generate(img_tensor, max_len=MAX_SEQ_LENGTH)
    # output_indices is a list of lists. For 1 image, output_indices[0].
    idx_list = output_indices

    result_words = []
    # end_token_idx = vocab.stoi["<end>"]
    end_token_idx = vocab.stoi["endofseq"]
    for idx in idx_list:
        if idx == end_token_idx:
            break
        # word = vocab.itos.get(idx, "<unk>")
        word = vocab.itos.get(idx, "unk")
        # skip <start>/<pad> in final output
        # if word not in ["<start>", "<pad>", "<end>"]:
        if word not in ["startofseq", "pad", "endofseq"]:
            result_words.append(word)
    return " ".join(result_words)


# -----------------------------------------------------------------
# 5. BUILD GRADIO INTERFACE
# -----------------------------------------------------------------
def main():
    iface = gr.Interface(
        fn=generate_caption_for_image,
        inputs=gr.Image(type="pil"),
        outputs="text",
        title="Image Captioning (ResNet + LSTM)",
        description="Upload an image to get a generated caption from the trained model.",
    )
    iface.launch(share=True)


if __name__ == "__main__":
    print("Loaded model. Starting Gradio interface...")
    main()
