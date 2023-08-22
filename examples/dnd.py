from lloom import LloomConfig, Lloom
import random

protagonist_name = "Storm Ryder"
storyteller_name = "Captain Quill"
quest = '''Set sail on the "Marauder's Dream" with your unique crew, seeking the fragmented map leading to the legendary "Mythic Isles" and the coveted "Seafarer's Heart" artifact. Battle fierce rivals, sea monsters, and unravel hidden histories while forging unbreakable bonds. A thrilling quest inspired by One Piece awaits with treasure beyond gold adventure, camaraderie, and freedom.'''
protagonist_description = "Storm Ryder, a daring and enigmatic adventurer, possesses the heart of a true pirate. Guided by an unyielding sense of justice and fueled by an insatiable thirst for adventure, Storm sails the uncharted seas, leaving a legacy of courage and camaraderie in their wake."
storyteller_description = "Captain Quill, a weathered and charismatic storyteller, carries tales of ancient legends and forgotten myths in their ink-stained logbook. With a twinkle in their eye and a voice that mesmerizes, they narrate the epic quest of Storm Ryder and the fabled Mythic Isles, inspiring awe and wonder in all who listen."

# I have already taken the ready-made system message from langchain docs, but the same functionality of so-called specifying the description can be implemented in 3 lines of code if needed
player_sysmsg = f'''
Here is the topic for a Dungeons & Dragons game: {quest}.
There is one player in this game: the protagonist, {protagonist_name}.
The story is narrated by the storyteller, {storyteller_name}.
Never forget you are the protagonist, {protagonist_name}, and I am the storyteller, {storyteller_name}. 
Your character description is as follows: {protagonist_description}.
You will propose actions you plan to take and I will explain what happens when you take those actions.
Speak in the first person from the perspective of {protagonist_name}.
For describing your own body movements, wrap your description in '*'.
Do not change roles!
Do not speak from the perspective of {storyteller_name}.
Do not forget to finish speaking by saying, 'It is your turn, {storyteller_name}.'
Do not add anything else.
Remember you are the protagonist, {protagonist_name}.
Stop speaking the moment you finish speaking from your perspective.
'''

master_sysmsg = f'''
Here is the topic for a Dungeons & Dragons game: {quest}.
There is one player in this game: the protagonist, {protagonist_name}.
The story is narrated by the storyteller, {storyteller_name}.
Never forget you are the storyteller, {storyteller_name}, and I am the protagonist, {protagonist_name}. 
Your character description is as follows: {storyteller_description}.
I will propose actions I plan to take and you will explain what happens when I take those actions. Some actions are complicated and you might ask me to roll a 20-sided dice to determine the outcome. 
I will give you the number and you will give me the outcome. Whenever I should roll a dice, you should end your phrase by saying "It is your turn, {protagonist_name}, roll a dice"
Speak in the first person from the perspective of {storyteller_name}.
For describing your own body movements, wrap your description in '*'.
Do not change roles!
Do not speak from the perspective of {protagonist_name}.
Do not forget to finish speaking by saying, 'It is your turn, {protagonist_name}.'
Do not add anything else.
Remember you are the storyteller, {storyteller_name}.
Stop speaking the moment you finish speaking from your perspective.
'''

master_config = LloomConfig(api_key="", temperature=1.0, logging=False, model="gpt-3.5-turbo-16k-0613",system_message=master_sysmsg)

master = Lloom(master_config)

max_iter = 6
initial_message = "I'm ready to start!"
n = 0

while n < max_iter:
    master_message = master.generate(initial_message)
    print(f"Master message: {master_message}")
    if "dice" in master_message:
        dice = random.randint(1, 20)
        print(f"You rolled a dice: {dice}")
        player_message = input() + f". The outcome of rolling a dice is {dice}"
        initial_message = player_message
    else:
        player_message = input()
        initial_message = player_message
    print(f"Player message: {initial_message}")
    n += 1
