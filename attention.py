#モデルの定義
import torch
import torch.nn as nn
import torch.optim as optim
from sudachipy import tokenizer
from sudachipy import dictionary

#from models import *
import pickle
import numpy as np

#辞書読み込み
with open('./static/model/id2word_m20_short.pkl', 'rb') as f:
    id2word = pickle.load(f)

with open('./static/model/word2id_m20_short.pkl', 'rb') as f:
    word2id = pickle.load(f)


# Encoderクラス
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, sequence):
        embedding = self.word_embeddings(sequence)
        # hsが各系列のGRUの隠れ層のベクトル
        # Attentionされる要素
        hs, h = self.gru(embedding)
        return hs, h

# Attention Decoderクラス
class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size):
        super(AttentionDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        # hidden_dim*2としているのは、各系列のGRUの隠れ層とAttention層で計算したコンテキストベクトルをtorch.catでつなぎ合わせることで長さが２倍になるため
        self.hidden2linear = nn.Linear(hidden_dim * 2, vocab_size)
        # 列方向を確率変換したいのでdim=1
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sequence, hs, h):
        embedding = self.word_embeddings(sequence)
        output, state = self.gru(embedding, h)

        # Attention層
        # bmmを使ってEncoder側の出力(hs)とDecoder側の出力(output)をbatchごとまとめて行列計算するために、Decoder側のoutputをbatchを固定して転置行列を取る
        t_output = torch.transpose(output, 1, 2) # t_output.size() = ([100, 128, 10])

        # bmmでバッチも考慮してまとめて行列計算
        s = torch.bmm(hs, t_output) # s.size() = ([100, 29, 10])

        # 列方向(dim=1)でsoftmaxをとって確率表現に変換
        # この値を後のAttentionの可視化などにも使うため、returnで返しておく
        attention_weight = self.softmax(s) # attention_weight.size() = ([100, 29, 10])

        # コンテキストベクトルをまとめるために入れ物を用意
        c = torch.zeros(self.batch_size, 1, self.hidden_dim, device=device) # c.size() = ([100, 1, 128])

        # 各DecoderのGRU層に対するコンテキストベクトルをまとめて計算する方法がわからなかったので、
        # 各層（Decoder側のGRU層は生成文字列が10文字なので10個ある）におけるattention weightを取り出してforループ内でコンテキストベクトルを１つずつ作成する
        # バッチ方向はまとめて計算できたのでバッチはそのまま
        for i in range(attention_weight.size()[2]): # 10回ループ

          # attention_weight[:,:,i].size() = ([100, 29])
          # i番目のGRU層に対するattention weightを取り出すが、テンソルのサイズをhsと揃えるためにunsqueezeする
          unsq_weight = attention_weight[:,:,i].unsqueeze(2) # unsq_weight.size() = ([100, 29, 1])

          # hsの各ベクトルをattention weightで重み付けする
          weighted_hs = hs * unsq_weight # weighted_hs.size() = ([100, 29, 128])

          # attention weightで重み付けされた各hsのベクトルをすべて足し合わせてコンテキストベクトルを作成
          weight_sum = torch.sum(weighted_hs, axis=1).unsqueeze(1) # weight_sum.size() = ([100, 1, 128])

          c = torch.cat([c, weight_sum], dim=1) # c.size() = ([100, i, 128])

        # 箱として用意したzero要素が残っているのでスライスして削除
        c = c[:,1:,:]

        output = torch.cat([output, c], dim=2) # output.size() = ([100, 10, 256])
        output = self.hidden2linear(output)
        return output, state, attention_weight

# Decoderのアウトプットのテンソルから要素が最大のインデックスを返す。つまり生成文字を意味する
def get_max_index(decoder_output, BATCH_NUM):
  results = []
  for h in decoder_output:
    results.append(torch.argmax(h))
  return torch.tensor(results, device=device).view(BATCH_NUM, 1)

#変数定義
input_len = 50
output_len = 100
embedding_dim = 200
hidden_dim = 128
BATCH_NUM=1
EPOCH_NUM = 50
vocab_size = len(word2id)
device = 'cpu'

# エンコーダーの設定
encoder = Encoder(vocab_size, embedding_dim, hidden_dim).to(device)
attn_decoder = AttentionDecoder(vocab_size, embedding_dim, hidden_dim, BATCH_NUM).to(device)

#学習済みモデルの読み込み
en_model_path = './static/model/en_model_cpu_m20_short_40'
de_model_path = './static/model/de_model_cpu_m20__short_40'

encoder.load_state_dict(torch.load(en_model_path))
attn_decoder.load_state_dict(torch.load(de_model_path))

# 分かち書きトークナイザーの設定
tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.C

def predict(text):
  #inputdataの作成
  input_data = np.zeros([1,input_len], dtype=int)
  #単語分割されたリストにする。
  wakati_text =  [m.surface() for m in tokenizer_obj.tokenize(text, mode)]
  if len(wakati_text)>=50:
        wakati_text = wakati_text[0:50]
  #入力を逆順にする。
  wakati_text.reverse()
  #知らない単語は<UNK>に置き換える
  for j in range(len(wakati_text)):
      word = wakati_text[j]
      if word in word2id:
          input_data[0][j] = word2id[word]
      else:
          input_data[0][j]  = word2id['<UNK>']

  #予測
  with torch.no_grad():
    input_tensor = torch.tensor(input_data, device=device)
    hs, encoder_state = encoder(input_tensor)

    # Decoderにはまず文字列生成開始を表す"_"をインプットにするので、"_"のtensorをバッチサイズ分作成
    start_char_batch = [[word2id["_"]] for _ in range(1)]
    decoder_input_tensor = torch.tensor(start_char_batch, device=device)

    decoder_hidden = encoder_state
    batch_tmp = torch.zeros(BATCH_NUM,1, dtype=torch.long, device=device)
    decoder_output, decoder_hidden, _ = attn_decoder(decoder_input_tensor, hs, decoder_hidden)
    # 予測文字を取得しつつ、そのまま次のdecoderのインプットとなる
    for _ in range(output_len - 1):
        decoder_output, decoder_hidden, _ = attn_decoder(decoder_input_tensor, hs, decoder_hidden)
        # 予測文字を取得しつつ、そのまま次のdecoderのインプットとなる
        decoder_input_tensor = get_max_index(decoder_output.squeeze().reshape(1,-1), BATCH_NUM)
        #decoder_input_tensor = get_max_index(decoder_output.squeeze())
        batch_tmp = torch.cat([batch_tmp, decoder_input_tensor], dim=1)

  return ''.join([id2word[i.item()] for i in batch_tmp[:,1:][0]]).replace('<pad>','')


if __name__ == "__main__":
  pass
  text = '日本のYouTuber、お笑いタレント、歌手、実業家。お笑いコンビ・オリエンタルラジオのボケ、ネタ作り担当。相方は藤森慎吾。ダンス&ボーカルグループ・RADIO FISHのメンバーとしても活動している。'
  out = predict(text)
  print(out)