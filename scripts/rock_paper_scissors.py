import random

def demo():
    keep_playing = True
    while keep_playing:
        plays = ["rock", "paper", "scissors"]
        computers_play = random.choice(plays)
        your_play = input("Input rock, paper, or scissors: ")
        while your_play != "rock" and your_play != "paper" and your_play != "scissors":
            your_play = input("Invalid input. Input rock, paper, or scissors: ")
        if your_play == computers_play:
            print("You played", your_play, "and the computer played", computers_play,
                  ". It's a tie.")
        elif your_play == "rock" and computers_play == "scissors":
            print("You played", your_play, " and the computer played", computers_play,
                  ". You win!")
        elif your_play == "rock" and computers_play == "paper":
            print("You played", your_play, " and the computer played", computers_play,
                  ". You lose.")
        elif your_play == "scissors" and computers_play == "paper":
            print("You played", your_play, " and the computer played", computers_play,
                  ". You win!")
        elif your_play == "scissors" and computers_play == "rock":
            print("You played", your_play, " and the computer played", computers_play,
                  ". You lose.")
        elif your_play == "paper" and computers_play == "scissors":
            print("You played", your_play, " and the computer played", computers_play,
                  ". You lose.")
        elif your_play == "paper" and computers_play == "rock":
            print("You played", your_play, " and the computer played", computers_play,
                  ". You win!")
        if input("Do you want to keep playing? If so, input 'y': ") != "y":
            keep_playing = False

if __name__ == "__main__":
    demo()