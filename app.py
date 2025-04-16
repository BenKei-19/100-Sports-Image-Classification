from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.init as init
from flask import send_from_directory

# Khởi tạo Flask
app = Flask(__name__)

# === Load model ===
class CNNModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(3,64,3,stride=1,padding='same'),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(64,64,3,stride=1,padding='same'),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(64,64,3,stride=1,padding='same'),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2,2)
        )
        self.skip_1 = nn.Sequential(
            nn.Conv2d(3,64,3,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(64,128,3,stride=1,padding='same'),
            nn.BatchNorm2d(128),
            nn.SiLU()
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(128,128,3,stride=1,padding='same'),
            nn.BatchNorm2d(128),
            nn.SiLU(),
        )
        self.conv_6 = nn.Sequential(
            nn.Conv2d(128,128,3,stride=1,padding='same'),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d(2,2),
        )
        self.skip_2 = nn.Sequential(
            nn.Conv2d(64,128,3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU()
        )

        self.conv_7 = nn.Sequential(
            nn.Conv2d(128,256,3,stride=1,padding='same'),
            nn.BatchNorm2d(256),
            nn.SiLU(),
        )
        self.conv_8 = nn.Sequential(
            nn.Conv2d(256,256,3,stride=1,padding='same'),
            nn.BatchNorm2d(256),
            nn.SiLU(),
        )
        self.conv_9 = nn.Sequential(
            nn.Conv2d(256,256,3,stride=1,padding='same'),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.MaxPool2d(2,2)
        )
        self.skip_3 = nn.Sequential(
            nn.Conv2d(128,256,3,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
        )

        self.conv_10 = nn.Sequential(
            nn.Conv2d(256,512,3,stride=1,padding='same'),
            nn.BatchNorm2d(512),
            nn.SiLU(),
        )
        self.conv_11 = nn.Sequential(
            nn.Conv2d(512,512,3,stride=1,padding='same'),
            nn.BatchNorm2d(512),
            nn.SiLU(),
        )
        self.conv_12 = nn.Sequential(
            nn.Conv2d(512,512,3,stride=1,padding='same'),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            nn.MaxPool2d(2,2)
        )
        self.skip_4 = nn.Sequential(
            nn.Conv2d(256,512,3,stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
        )

        self.avgpool = nn.AvgPool2d(kernel_size=14)

        self.dense_1 = nn.Sequential(
            nn.Linear(512,256),
            nn.SiLU(),
            nn.Dropout(p=0.5)
        )
        self.dense_3 = nn.Sequential(
            nn.Linear(256,n_classes)
        )
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
          if isinstance(m,nn.Conv2d):
            init.kaiming_normal_(m.weight,nonlinearity='linear')
            if m.bias is not None:
              init.zeros_(m.bias)
          elif isinstance(m,nn.Linear):
            init.kaiming_normal_(m.weight,nonlinearity='linear')
            if m.bias is not None:
              init.zeros_(m.bias)

    def forward(self, x):
        prev = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        res = self.skip_1(prev)
        x = x + res

        prev = x
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        res = self.skip_2(prev)
        x = x + res

        prev = x
        x = self.conv_7(x)
        x = self.conv_8(x)
        x = self.conv_9(x)
        res = self.skip_3(prev)
        x = x + res

        prev = x
        x = self.conv_10(x)
        x = self.conv_11(x)
        x = self.conv_12(x)
        res = self.skip_4(prev)
        x = res + x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dense_1(x)
        x = self.dense_3(x)
        return x
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel(n_classes=100).to(device)
state_dict = torch.load('./Models/best_model.pth', map_location=device)
new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
model.load_state_dict(new_state_dict)
model.eval()
class_names = ['air hockey', 'ampute football', 'archery', 'arm wrestling', 'axe throwing', 'balance beam', 'barell racing', 'baseball', 'basketball', 'baton twirling', 'bike polo', 'billiards', 'bmx', 'bobsled', 'bowling', 'boxing', 'bull riding', 'bungee jumping', 'canoe slamon', 'cheerleading', 'chuckwagon racing', 'cricket', 'croquet', 'curling', 'disc golf', 'fencing', 'field hockey', 'figure skating men', 'figure skating pairs', 'figure skating women', 'fly fishing', 'football', 'formula 1 racing', 'frisbee', 'gaga', 'giant slalom', 'golf', 'hammer throw', 'hang gliding', 'harness racing', 'high jump', 'hockey', 'horse jumping', 'horse racing', 'horseshoe pitching', 'hurdles', 'hydroplane racing', 'ice climbing', 'ice yachting', 'jai alai', 'javelin', 'jousting', 'judo', 'lacrosse', 'log rolling', 'luge', 'motorcycle racing', 'mushing', 'nascar racing', 'olympic wrestling', 'parallel bar', 'pole climbing', 'pole dancing', 'pole vault', 'polo', 'pommel horse', 'rings', 'rock climbing', 'roller derby', 'rollerblade racing', 'rowing', 'rugby', 'sailboat racing', 'shot put', 'shuffleboard', 'sidecar racing', 'ski jumping', 'sky surfing', 'skydiving', 'snow boarding', 'snowmobile racing', 'speed skating', 'steer wrestling', 'sumo wrestling', 'surfing', 'swimming', 'table tennis', 'tennis', 'track bicycle', 'trapeze', 'tug of war', 'ultimate', 'uneven bars', 'volleyball', 'water cycling', 'water polo', 'weightlifting', 'wheelchair basketball', 'wheelchair racing', 'wingsuit flying']

# Tiền xử lý ảnh
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4713, 0.4699, 0.4548], std=[0.2931, 0.2850, 0.2986]),
])


@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory('Images', filename)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    image_file = request.files['imagefile']
    filename = image_file.filename
    image_path = './Images/' + filename
    image_file.save(image_path)


    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    prediction = class_names[predicted.item()]

    return render_template('index.html', prediction=prediction, image_url='/images/' + filename)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
