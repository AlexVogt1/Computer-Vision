'''
Matthew Dacre: 2091295
Joshua Wacks: 2143116
Alex Vogt: 2152320
'''

from classes import *
i = 0
puzzle = Puzzle(MATCH_IMGS)
corner_piece = puzzle.pieces[3]
# Start BFS by adding in the bottom left corner piece
queue = []
queue.append(corner_piece)
corner_piece.insert(count=i)
i += 1
corner_piece.inserted = True

# TODO: Rest of BFS

while queue:
	curr = queue.pop(0)

	for edge in curr.return_edge():
		if edge == None:
			break
		if edge.connected_edge:
			next_piece = edge.connected_edge.parent_piece
			if not next_piece.inserted:
				queue.append(next_piece)
				next_piece.insert(count=i)
				i += 1
				next_piece.inserted = True