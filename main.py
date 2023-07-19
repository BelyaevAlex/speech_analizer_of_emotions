from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
from transformers import BertForSequenceClassification, AutoTokenizer
import torchaudio
import torch
import os
import openai
from pydub import AudioSegment
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor


bot = Bot(token="token")
dp = Dispatcher(bot)


def analizing_all(filename):
                openai.api_key = "api_key"
                audio_file = open(filename, "rb")
                print(audio_file)
                transcript = openai.Audio.transcribe("whisper-1", audio_file)
                text = transcript["text"]
                print(text)
                response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user",
                                   "content": f"Decide whether a Tweet's sentiment is positive, neutral, sad or angry.\n\nTweet: \n'{text}'\nSentiment:"}]
                )
                print(1)
                num2emotion = {0: '😐(нейтральная)', 1: '😡(злая)', 2: '😀(позитивная)', 3: '😢(грусная)', 4: '😐(нейтральная)'}
                emotion2num = {'neutral': 0, 'angry': 1, 'positive': 2, 'sad': 3, 'other': 4}

                result_speech = analizing_speech(filename)
                print(2)
                print(response)
                if len(response.choices[0].message.content.split(' ')) > 1:
                        raise Exception("Incorrect result of chatgpt")
                if result_speech[emotion2num.get(response.choices[0].message.content.lower())] >= 0.5:
                        return text, f"по произншению и содержанию: {num2emotion.get(emotion2num.get(response.choices[0].message.content.lower()))}"
                        print(2)
                else:
                        return text, f"по произншению: {num2emotion.get(result_speech.index(max(result_speech)))}, а по содержанию: {num2emotion.get(emotion2num.get(response.choices[0].message.content.lower()))}"
                        print(3)
        
        


def analizing_speech(filepath):
                feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("./facebook/hubert-large-ls960-ft")
                model = HubertForSequenceClassification.from_pretrained("./xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned")
                num2emotion = {0: 'нейтральная', 1: 'злая', 2: 'позитивная', 3: 'грусная', 4: 'другая'}
                print(3)
                waveform, sample_rate = torchaudio.load(f"{filepath.split('.')[0]}.wav")
                transform = torchaudio.transforms.Resample(sample_rate, 16000)
                print(4)
                waveform = transform(waveform)
                inputs = feature_extractor(
                        waveform,
                        sampling_rate=feature_extractor.sampling_rate,
                        return_tensors="pt",
                        padding=True,
                        max_length=16000 * 10,
                        truncation=True
                    )
                print(5)
                logits = model(inputs['input_values'][0]).logits
                print(6)
                predictions = torch.argmax(logits, dim=-1)
                predicted_emotion = num2emotion[predictions.numpy()[0]]
                return(logits[0].tolist())


def split_audio(input_prefix):
    AudioSegment.from_wav(f"{input_prefix}.wav").export(f"{input_prefix}.mp3", format="mp3")
    audio = AudioSegment.from_file(f"{input_prefix}.mp3")

    # Подсчитываем общую длительность аудио в миллисекундах
    total_duration = len(audio)
    if total_duration > 10000:
            chunk = audio[0:5000]
            output_file = f"{input_prefix}_start.mp3"
            chunk.export(output_file, format="mp3")
            AudioSegment.from_mp3(f"{output_file}").export(f"{input_prefix}_start.wav", format="wav")
            chunk = audio[total_duration-5000:]
            output_file = f"{input_prefix}_end.mp3"
            chunk.export(output_file, format="mp3")
            AudioSegment.from_mp3(f"{output_file}").export(f"{input_prefix}_end.wav", format="wav")
            return 1
    else:
            return 0


def start(input_file):
        a = split_audio(input_file)
        if a == 1:
                text_start, result_start = analizing_all(f'{input_file}_start.mp3')
                text_end, result_end = analizing_all(f'{input_file}_end.mp3')
                if result_start == "error" or result_end == "error":
                        return 'error', 'error', 'error', 'error'
                else:
                        return result_start, result_end, text_start, text_end
        else:
                return 'error', 'error', 'error', 'error'


@dp.message_handler(content_types=['voice'])
async def voice_processing(message):
    await message.voice.download("test10.ogg")
    AudioSegment.from_ogg("test10.ogg").export("test10.wav", format="wav")
    result_start, result_end, text_start, text_end = start("test10")
    if result_start == 'error':
            await bot.send_message(message.from_user.id, f'Что-то пошло не так, попробуйте снова, возможно проблема в том, что аудио длинной менее 20 секунд')
    else:
            await bot.send_message(message.from_user.id, f'Текст в начале: {text_start}')
            await bot.send_message(message.from_user.id, f'В начале {result_start}')
            await bot.send_message(message.from_user.id, f'Текст в конце: {text_end}')
            await bot.send_message(message.from_user.id, f'В конце {result_end}')


@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    await message.reply("Привет!\nОтправь мне голосовое сообщение не короче 20 секунд и я определю твою эмоцию в начале голосового сообщения и в конце")


if __name__ == '__main__':
    executor.start_polling(dp)
