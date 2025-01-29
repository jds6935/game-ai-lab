from pathlib import Path
import sys
import time
sys.path.append(str(Path(__file__).parents[1]))

from ollama import chat
from util.llm_utils import pretty_stringify_chat, ollama_seed as seed

# Add you code below
sign_your_name = 'Joel Sander'
model = 'deepseek-r1:14b'
messages = [
  {'role': 'system', 'content': 'You are a dungeon master. You need to interact with the players and give them options to choose from. From the players chosen option, create a response that will further the story. \
    each response you give should end in more options for the players to choose from. You can also give the players information about the world they are in. Do not make a whole story, \
    but rather a series of interactions that will lead to a story. You can also ask the players questions to further the story. your reponse shoudld be in the form of a (one) paragraph, followed by a list of options for the players to choose from. \
    Do not make choices for the player, the player can also with to do something that is not in the options you give them. If the player does something that is not in the options, \
    you can respond to that as well. you reserve the right to refuse player actions that are not in the options you gave them'},
  {'role': 'assistant', 'content': '<PARAGRAPH>(a short paragraph giving exposition, story, new information, and the effects of previous player actions)</PARAGRAPH>\n\nOptions:\n1. <OPTION>\n2. <OPTION>\n3. <OPTION> \n What do you do?'},

]
options = {'temperature': 0.2, 'max_tokens': 50}


# But before here.

options |= {'seed': seed(sign_your_name + str(int(time.time())))}
# Chat loop
while True:
  response = chat(model=model, messages=messages, stream=False, options=options)
  # Add your code below
  print(f'Agent: {response.message.content}')
  messages.append({'role': 'assistant', 'content': response.message.content})

  message = {'role': 'user', 'content': input('You: ')}
  messages.append(message)

  # But before here.
  if messages[-1]['content'] == '/exit':
    break

# Save chat
with open(Path('lab03/attempts.txt'), 'a') as f:
  file_string  = ''
  file_string +=       '-------------------------NEW ATTEMPT-------------------------\n\n\n'
  file_string += f'Model: {model}\n'
  file_string += f'Options: {options}\n'
  file_string += pretty_stringify_chat(messages)
  file_string += '\n\n\n------------------------END OF ATTEMPT------------------------\n\n\n'
  f.write(file_string)

