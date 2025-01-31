## Prompt Engineering Process

### attempts with prompts and responses are in attempts.txt

### Attempt 1
#### Intention
> Get the model to respond at all.

#### Action/Change
> Used a distilled version of the DeepSeek model.

#### Result
> Successfully got the model to respond.

#### Reflection/Analysis of the result
> The system is RAM-limited, which was not recorded in attempts.txt.

---

### Attempt 2
#### Intention
> Make the model behave as a Dungeons & Dragons (D&D) Dungeon Master.

#### Action/Change
> Modified the system prompt to set the model as a Dungeon Master.

#### Result
> The model generated a full D&D campaign story instead of engaging interactively.

#### Reflection/Analysis of the result
> The model did not engage dynamically but instead provided a predetermined story. Further prompt engineering is required.

---

### Attempt 3
#### Intention
> Encourage interactive storytelling by giving players choices.

#### Action/Change
> Adjusted the prompt to guide the model towards providing choices within its narrative.

#### Result
> The model still generated a full story but started incorporating choices for the players within the narrative.

#### Reflection/Analysis of the result
> Some progress was made, but the model still lacks direct user engagement.

---

### Attempt 4
#### Intention
> Ensure the model prompts the player directly for decisions.

#### Action/Change
> Tweaked system prompt further.

#### Result
> No progress in the intended direction.

#### Reflection/Analysis of the result
> The model still focuses on storytelling rather than interaction.

---

### Attempt 5
#### Intention
> Make the model explicitly ask the player for a decision.

#### Action/Change
> Prompted the model to provide options but also require a player response.

#### Result
> The model evaluated its own response instead of asking for player input.

#### Reflection/Analysis of the result
> The model followed the system prompt but rationalized its choices instead of prompting the user.

---

### Attempt 6
#### Intention
> Force the model to stop generating full stories and ask the user for input.

#### Action/Change
> Removed "Your purpose is to interact with the user to create and tell a vibrant story for the players to enjoy." from the system prompt.

#### Result
> The model still generated full stories, though it sometimes included choices.

#### Reflection/Analysis of the result
> The phrasing of the system prompt still needs adjustment to reinforce user interaction.

---

### Attempt 7
#### Intention
> Improve performance by running the model on a stronger system.

#### Action/Change
> Moved to an RTX 2060 system and switched to DeepSeek-r1:14b.

#### Result
> Significant improvement. The model mostly provided exposition followed by player choices but occasionally reverted to making choices for the player.

#### Reflection/Analysis of the result
> Larger model improved coherence and interaction, but stability over multiple prompts remains an issue.

---

### Attempt 8
#### Intention
> Ensure the model does not pick choices for the player.

#### Action/Change
> Further refined prompt engineering.

#### Result
> The model consistently provided choices and allowed the user to input custom choices.

#### Reflection/Analysis of the result
> A major step in the right direction. The model finally engaged interactively.

---

### Attempt 9
#### Intention
> Observe natural behavior of the model without additional prompt changes.

#### Action/Change
> Introduced Unix time as a seed to vary responses.

#### Result
> Success. The model demonstrated variability in responses, denying unrealistic player actions while allowing meaningful consequences.

#### Reflection/Analysis of the result
> Adding randomness improved realism. The model could reject bad actions without making the game unfair.

---

### Attempt 10
#### Intention
> Test one-shot learning on a smaller model.

#### Action/Change
> Switched to the 1.5B parameter model and applied refined prompting.

#### Result
> The model repeatedly answered its own prompts instead of interacting.

#### Reflection/Analysis of the result
> The 1.5B model lacks the capacity for complex interactions, possibly requiring further tuning.

---

### Attempt 11
#### Intention
> Return to a larger model for better performance.

#### Action/Change
> Switched back to the 14B model.

#### Result
> The model worked better, reinforcing the idea that parameter count significantly impacts performance.

#### Reflection/Analysis of the result
> The 1.5B model may be too small for this application. Further prompt engineering might help, but hardware limitations exist.