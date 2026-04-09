import os
import random
from PIL import Image, ImageDraw

os.makedirs("data/images", exist_ok=True)

positions = [
    ["top left","top middle","top right"],
    ["middle left","center","middle right"],
    ["bottom left","bottom middle","bottom right"]
]

def generate_board():
    board = [["" for _ in range(3)] for _ in range(3)]
    num_pieces = random.randint(2,4)

    for _ in range(num_pieces):
        i, j = random.randint(0,2), random.randint(0,2)
        board[i][j] = random.choice(["X","O"])

    return board

def board_to_sentence(board):
    parts = []
    for i in range(3):
        for j in range(3):
            if board[i][j] != "":
                parts.append(f"{board[i][j]} in {positions[i][j]}")
    return ", ".join(parts)

def draw_board(board, filename):
    img = Image.new("RGB", (128,128), "white")
    draw = ImageDraw.Draw(img)

    cell = 128 // 3

    # grid
    for i in range(1,3):
        draw.line((0, i*cell, 128, i*cell), fill="black")
        draw.line((i*cell, 0, i*cell, 128), fill="black")

    # pieces
    for i in range(3):
        for j in range(3):
            x = j*cell + cell//2
            y = i*cell + cell//2

            if board[i][j] == "X":
                draw.text((x-10,y-10),"X", fill="black")
            elif board[i][j] == "O":
                draw.text((x-10,y-10),"O", fill="black")

    img.save(filename)

def board_to_label(board):
    # 0 empty, 1 X, 2 O
    label = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == "":
                label.append(0)
            elif board[i][j] == "X":
                label.append(1)
            else:
                label.append(2)
    return label

def main():
    with open("data/labels.txt","w") as f:
        for i in range(2000):
            board = generate_board()
            sentence = board_to_sentence(board)
            label = board_to_label(board)

            draw_board(board, f"data/images/{i}.png")

            label_str = " ".join(map(str,label))
            f.write(f"{i}.png\t{label_str}\t{sentence}\n")

if __name__ == "__main__":
    main()