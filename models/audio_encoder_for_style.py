import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as transforms

class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              bias=bias)

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

class Encoder_lf0(nn.Module):
    def __init__(self, typ='no_emb'):
        super(Encoder_lf0, self).__init__()
        self.type = typ
        if typ != 'no_emb':
            convolutions = []
            for i in range(3):
                conv_layer = nn.Sequential(
                    ConvNorm(1 if i == 0 else 256, 256,
                             kernel_size=5, stride=2 if i == 2 else 1,
                             padding=2,
                             dilation=1, w_init_gain='relu'),
                    nn.GroupNorm(256 // 16, 256),
                    nn.ReLU())
                convolutions.append(conv_layer)
            self.convolutions = nn.ModuleList(convolutions)
            self.lstm = nn.LSTM(256, 32, 1, batch_first=True, bidirectional=True)

    def forward(self, lf0):
        if self.type != 'no_emb':
            if len(lf0.shape) == 2:
                lf0 = lf0.unsqueeze(1)  # bz x 1 x 128
            for conv in self.convolutions:
                lf0 = conv(lf0)  # bz x 256 x 128
            lf0 = lf0.transpose(1, 2)  # bz x 64 x 256
            self.lstm.flatten_parameters()
            lf0, _ = self.lstm(lf0)  # bz x 64 x 64
        else:
            if len(lf0.shape) == 2:
                lf0 = lf0.unsqueeze(-1)  # bz x 128 x 1 # no downsampling
        return lf0

def pad_layer(inp, layer, pad_type='reflect'):
    kernel_size = layer.kernel_size[0]
    if kernel_size % 2 == 0:
        pad = (kernel_size // 2, kernel_size // 2 - 1)
    else:
        pad = (kernel_size // 2, kernel_size // 2)
    inp = F.pad(inp, pad=pad, mode=pad_type)
    out = layer(inp)
    return out

def conv_bank(x, module_list, act, pad_type='reflect'):
    outs = []
    for layer in module_list:
        out = act(pad_layer(x, layer, pad_type))
        outs.append(out)
    out = torch.cat(outs + [x], dim=1)
    return out

def get_act(act):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'lrelu':
        return nn.LeakyReLU()
    else:
        return nn.ReLU()

class SpeakerEncoder(nn.Module):
    def __init__(self, c_in=80, c_h=128, c_out=256, kernel_size=5,
                 bank_size=8, bank_scale=1, c_bank=128,
                 n_conv_blocks=6, n_dense_blocks=6,
                 subsample=[1, 2, 1, 2, 1, 2], act='relu', dropout_rate=0):
        super(SpeakerEncoder, self).__init__()
        self.c_in = c_in
        self.c_h = c_h
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.n_conv_blocks = n_conv_blocks
        self.n_dense_blocks = n_dense_blocks
        self.subsample = subsample
        self.act = get_act(act)
        self.conv_bank = nn.ModuleList(
            [nn.Conv1d(c_in, c_bank, kernel_size=k) for k in range(bank_scale, bank_size + 1, bank_scale)])
        in_channels = c_bank * (bank_size // bank_scale) + c_in
        self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)
        self.first_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size) for _ in range(n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub)
                                                 for sub, _ in zip(subsample, range(n_conv_blocks))])
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.first_dense_layers = nn.ModuleList([nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)])
        self.second_dense_layers = nn.ModuleList([nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)])
        self.output_layer = nn.Linear(c_h, c_out)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def conv_blocks(self, inp):
        out = inp
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.subsample[l] > 1:
                out = F.avg_pool1d(out, kernel_size=self.subsample[l], ceil_mode=True)
            out = y + out
        return out

    def dense_blocks(self, inp):
        out = inp
        for l in range(self.n_dense_blocks):
            y = self.first_dense_layers[l](out)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = self.second_dense_layers[l](y)
            y = self.act(y)
            y = self.dropout_layer(y)
            out = y + out
        return out

    def forward(self, x):
        out = conv_bank(x, self.conv_bank, act=self.act)
        out = pad_layer(out, self.in_conv_layer)
        out = self.act(out)
        out = self.conv_blocks(out)
        out = self.pooling_layer(out).squeeze(2)
        out = self.dense_blocks(out)
        out = self.output_layer(out)
        return out
    
    def initialize_weights(self):
        def init_func(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        self.apply(init_func)

# Define the MelSpectrogram and AmplitudeToDB transforms
mel_spectrogram_transform = transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    win_length=400,
    hop_length=160,
    n_mels=80
).to('cuda')

amplitude_to_db_transform = transforms.AmplitudeToDB()

def extract_logmel_torchaudio_tensor(audio_tensor, sr=16000):
    """
    Extract log mel spectrogram from a given audio tensor.

    Args:
    audio_tensor (torch.Tensor): Input audio tensor of shape (batch_size, num_samples).
    sr (int): Sample rate of the audio.

    Returns:
    torch.Tensor: Log mel spectrogram tensor of shape (batch_size, n_mels, time).
    """
    mel_spectrogram = mel_spectrogram_transform(audio_tensor)
    log_mel_spectrogram = amplitude_to_db_transform(mel_spectrogram)
    return log_mel_spectrogram

if __name__ == '__main__':
    # Example usage
    batch_size = 1
    sample_rate = 16000
    audio_length = int(5 * sample_rate)  # 4.5 seconds of audio
    audio_batch = torch.randn(batch_size, audio_length)  # Simulate a batch of audio samples

    log_mel_spectrograms = extract_logmel_torchaudio_tensor(audio_batch, sr=sample_rate)

    # Prepare input for the SpeakerEncoder
    # log_mel_spectrograms = log_mel_spectrograms.permute(0, 2, 1)  # (batch_size, n_mels, time)

    # Initialize the SpeakerEncoder
    c_in = log_mel_spectrograms.size(1)
    print(log_mel_spectrograms.shape)
    print(c_in)
    speaker_encoder = SpeakerEncoder(c_in=c_in)

    # Forward pass to get the speaker embeddings
    speaker_embeddings = speaker_encoder(log_mel_spectrograms)

    print("Speaker Embeddings shape:", speaker_embeddings.shape)
