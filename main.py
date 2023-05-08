import telebot
import requests


token = "6187935126:AAFPZdUzeVY_jPDPG_sCAe0GVObCPL7AQ84"

bot = telebot.TeleBot(token)


@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):
    bot.reply_to(message, "Howdy, how are you doing?")


@bot.message_handler(commands=['description'])
def send_description(message):
    bot.reply_to(message,
                 "A chat bot using natural language processing (NLP) and Python is a project that involves creating a "
                 "software program that can simulate human-like conversation with users through a messaging "
                 "interface. The chat bot uses NLP techniques to understand and respond to the user's messages in a "
                 "natural way.")


# @bot.message_handler(func=lambda msg: True)
# def echo_all(message):
# bot.reply_to(message, message.text)


@bot.message_handler(commands=['pic'])
def send_pic(message):
    print("send_pic function called")  # Print statement to check if the function is being called properly
    # Fetch a random cat image from the Cat API
    response = requests.get('https://api.thecatapi.com/v1/images/search')
    response_json = response.json()
    image_url = response_json[0]['url']
    # Download the image and send it to the user
    response = requests.get(image_url)
    photo = response.content
    print("photo variable defined")  # Print statement to check if the photo variable is properly defined
    bot.send_photo(message.chat.id, photo)


# Start the bot and listen for incoming messages
bot.polling()
