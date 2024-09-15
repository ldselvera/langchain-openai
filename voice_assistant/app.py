from dotenv import load_dotenv
load_dotenv()

from interface import AudioInterface
from agents import ConversationAgent, SmartChatAgent

interface = AudioInterface()
agent = ConversationAgent()
# agent = SmartChatAgent()

while True:
    text = interface.listen()
    response = agent.run(text)
    interface.speak(response)


