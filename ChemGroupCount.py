import numpy as np
import argparse

class Atom:
    def __init__(self, symbol, position, index):
        self.symbol = symbol
        self.position = np.array(position)
        self.index = index

def read_structure(file_name):
    atoms = []
    with open(file_name, 'r') as file:
        lines = file.readlines()
        scale_factor = float(lines[1].strip())
        cell = [np.array(lines[i].split(), dtype=float) * scale_factor for i in range(2, 5)]
        symbols = lines[5].split()
        num_atoms = [int(x) for x in lines[6].split()]
        positions = lines[8:8+sum(num_atoms)]
        
        atom_index = 0
        for symbol, num in zip(symbols, num_atoms):
            for i in range(num):
                position = np.array(positions[atom_index].split(), dtype=float)
                atoms.append(Atom(symbol, position, atom_index))
                atom_index += 1

    return atoms, cell

def apply_pbc(atom1, atom2, cell):
    min_image_dist = np.inf
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            for z in [-1, 0, 1]:
                translation = x*cell[0] + y*cell[1] + z*cell[2]
                dist = np.linalg.norm(atom1.position + translation - atom2.position)
                if dist < min_image_dist:
                    min_image_dist = dist
    return min_image_dist

def find_neighbors(atom, atoms, cutoff, cell):
    neighbors = []
    for other_atom in atoms:
        if atom != other_atom:
            distance = apply_pbc(atom, other_atom, cell)
            if distance <= cutoff:
                neighbors.append(other_atom)
    return neighbors

def main():
    parser = argparse.ArgumentParser(description="Process a structure file.")
    parser.add_argument("file_name", type=str, help="Name of the structure file")
    parser.add_argument("--oh_cutoff", type=float, default=1.2, help="Cutoff distance for O-H bond")
    parser.add_argument("--ch_cutoff", type=float, default=1.2, help="Cutoff distance for C-H bond")
    args = parser.parse_args()

    atoms, cell = read_structure(args.file_name)
    oh_cutoff = args.oh_cutoff
    ch_cutoff = args.ch_cutoff




    water_molecules = []
    hydroxyl_groups = []
    isolated_atoms = []
    ch_groups = []
    ch2_groups = []
    ch3_groups = []
    isolated_carbons = []

    atom_in_group = set()

    for atom in atoms:
        if atom.symbol == 'C' and atom.index not in atom_in_group:
            hydrogen_neighbors = [a for a in find_neighbors(atom, atoms, 1.2, cell) if a.symbol == 'H']
            num_hydrogens = len(hydrogen_neighbors)

            if num_hydrogens == 1:
                ch_groups.append((atom, hydrogen_neighbors))
                atom_in_group.add(atom.index)
                atom_in_group.update([h.index for h in hydrogen_neighbors])
            elif num_hydrogens == 2:
                ch2_groups.append((atom, hydrogen_neighbors))
                atom_in_group.add(atom.index)
                atom_in_group.update([h.index for h in hydrogen_neighbors])
            elif num_hydrogens == 3:
                ch3_groups.append((atom, hydrogen_neighbors))
                atom_in_group.add(atom.index)
                atom_in_group.update([h.index for h in hydrogen_neighbors])
            elif num_hydrogens == 4:
                ch4_groups.append((atom, hydrogen_neighbors))   
            else:
                isolated_carbons.append(atom)

    for atom in atoms:
        if atom.symbol == 'O':
            neighbors = find_neighbors(atom, atoms, cutoff_H_O, cell)
            hydrogen_neighbors = [a for a in neighbors if a.symbol == 'H']
            
            if len(hydrogen_neighbors) == 2:
                water_molecules.append((atom, hydrogen_neighbors))
                atom_in_group.add(atom.index)
                atom_in_group.update([h.index for h in hydrogen_neighbors])
            elif len(hydrogen_neighbors) == 1:
                hydroxyl_groups.append((atom, hydrogen_neighbors))
                atom_in_group.add(atom.index)
                atom_in_group.update([h.index for h in hydrogen_neighbors])

    for atom in atoms:
        if atom.index not in atom_in_group:
            isolated_atoms.append(atom)


    print(f"Number of CH groups: {len(ch_groups)}")
    for i, group in enumerate(ch_groups):
        atom_indices = [group[0].index] + [atom.index for atom in group[1]]
        print(f"CH group {i}: Atom indices {atom_indices}")

    print(f"Number of CH2 groups: {len(ch2_groups)}")
    for i, group in enumerate(ch2_groups):
        atom_indices = [group[0].index] + [atom.index for atom in group[1]]
        print(f"CH2 group {i}: Atom indices {atom_indices}")

    print(f"Number of CH3 groups: {len(ch3_groups)}")
    for i, group in enumerate(ch3_groups):
        atom_indices = [group[0].index] + [atom.index for atom in group[1]]
        print(f"CH3 group {i}: Atom indices {atom_indices}")
    print(f"Number of CH4 groups: {len(ch4_groups)}")
    for i, group in enumerate(ch4_groups):
        atom_indices = [group[0].index] + [atom.index for atom in group[1]]
        print(f"CH4 group {i}: Atom indices {atom_indices}")

    print(f"Number of isolated carbon atoms: {len(isolated_carbons)}")
    for atom in isolated_carbons:
        print(f"Isolated carbon atom at index {atom.index}")

    print(f"Number of water molecules: {len(water_molecules)}")
    for i, group in enumerate(water_molecules):
        atom_indices = [group[0].index] + [atom.index for atom in group[1]]
        print(f"Water molecule {i}: Atom indices {atom_indices}")


    print(f"Number of hydroxyl groups: {len(hydroxyl_groups)}")
    for i, group in enumerate(hydroxyl_groups):
        atom_indices = [group[0].index] + [atom.index for atom in group[1]]
        print(f"Hydroxyl group {i}: Atom indices {atom_indices}")


#    print(f"Number of water molecules: {len(water_molecules)}")
#    for i, group in enumerate(water_molecules):
#        print(f"Water molecule {i}: Atom indices {[atom.index for atom in group]}")


    print(f"Number of isolated atoms: {len(isolated_atoms)}")
    for atom in isolated_atoms:
        print(f"Isolated atom: {atom.symbol} at index {atom.index}")

if __name__ == "__main__":
    main()

