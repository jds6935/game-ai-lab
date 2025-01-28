## Prompt Engineering Process
### attempts with prompts and responses are in attempts.txt

1. I decided to use a distilled verion of the deepseek model. My intial attempts were to get the model to respond at all. I was successful. The system is ram limited (not in attempts.txt)
2. My next attempt was to change the system prompt do tell the model it is a dnd dungeon master. I need to prompt engineer as it just gave me a whole story of dnd campaign.
3. Still gave me a whole story but gave the players in the story it told options.
4. No progress in the right direction
5. Gave options, but didn't directly ask the player to choose one. instead it evaluated itself based on the system prompt and justified why it responded in such a way.
6. Told it to give a paragraph and a list of options, still not prompting the user. Making whole story. will remove: "Your purpose is to interact with the user to create and tell a vibrant story for the players \
    to enjoy." in next stages.
7. This Attempt did not save to the txt. Moved the program to my system with an RTX 2060 and changed the model to deepseek-r1:14b. This greatly moved the results towards the direction wanted. The model now, for the most part only creates some exposition and gives the player choices to do. after a few requests, the model reverted to making choices for the player.
8. This attempt is very good and gave the user choices with picking it for them. allowed the user to also type in other options not in the list