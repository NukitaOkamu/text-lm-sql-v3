#!/usr/bin/env python3

# split_save_pkg.py
# This script initiates the text-to-SQL pipeline by parsing an Oracle PL/SQL package body file 
# and splitting its procedures and functions into individual .SQL files. 
# It uses regex to identify routine boundaries, tokenizes "begin" and "end" blocks 
# to determine scope, and saves each routine in a schema/package folder structure within the 
# specified output directory (e.g., out/), enabling subsequent metadata extraction.
import re
import sys
import os

def remove_line_comment(line):
    # Remove any -- comments from the line.
    return re.sub(r'--.*$', '', line)

def tokenize_line(line):
    """
    Returns a list of tokens ("begin" or "end") found in the line.
    Recognizes "begin", "end" (standalone or named), and "end loop" as block delimiters.
    """
    tokens = []
    pattern = re.compile(
        r'\b(begin)\b|\bend\b(?!\s+(if|loop|case))(?=\s*(;|\w+|$))|\bend\s+loop\b',
        re.IGNORECASE
    )
    for m in pattern.finditer(line):
        if m.group(1):  # "begin"
            tokens.append("begin")
        elif m.group(0).lower().startswith("end loop"):
            tokens.append("end")
        else:  # "end" (standalone or named)
            tokens.append("end")
    return tokens

def parse_routines(file_path):
    """
    Parses an Oracle package body file and returns a tuple (routines, lines).
    Each routine is represented as a dictionary with keys:
      - 'type': 'procedure' or 'function'
      - 'name': routine name
      - 'start': line number (1-indexed) where the routine header appears
      - 'end': line number (1-indexed) where the outermost block ends (or None if not found)
      - 'schema': schema name from the most recent package header (if any)
      - 'package': package name from the most recent package header
    """
    routines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    pkg_pattern = re.compile(r'CREATE\s+OR\s+REPLACE\s+PACKAGE\s+BODY\s+((\w+)\.)?(\w+)', re.IGNORECASE)
    header_pattern = re.compile(r'^\s*(procedure|function)\s+(\w+)', re.IGNORECASE)
    external_lang_pattern = re.compile(r'\b(is|as)\s+language\s+(java|c)\b', re.IGNORECASE)

    current_schema = None
    current_package = None
    i = 0
    while i < len(lines):
        line = lines[i]
        pkg_match = pkg_pattern.search(line)
        if pkg_match:
            current_schema = pkg_match.group(2)
            current_package = pkg_match.group(3)
        header_match = header_pattern.match(line)
        if header_match:
            routine_type = header_match.group(1).lower()
            routine_name = header_match.group(2)
            routine_start = i + 1
            body_started = False
            nest = 0
            routine_end = None
            is_external = False
            j = i + 1
            while j < len(lines):
                current_line = remove_line_comment(lines[j]).strip()
                if external_lang_pattern.search(current_line):
                    is_external = True
                    while j < len(lines) and ';' not in remove_line_comment(lines[j]):
                        j += 1
                    routine_end = j + 1 if j < len(lines) else len(lines)
                    break
                elif re.search(r'\b(begin|is|as)\b', current_line, re.IGNORECASE):
                    break
                j += 1
            if is_external:
                i = j
                continue
            while j < len(lines):
                current_line = remove_line_comment(lines[j])
                if not body_started:
                    if re.search(r'\bbegin\b', current_line, flags=re.IGNORECASE):
                        body_started = True
                        tokens = tokenize_line(current_line)
                        for token in tokens:
                            if token == "begin":
                                nest += 1
                            elif token == "end":
                                nest -= 1
                        if nest <= 0:
                            routine_end = j + 1
                            break
                else:
                    tokens = tokenize_line(current_line)
                    for token in tokens:
                        if token == "begin":
                            nest += 1
                        elif token == "end":
                            nest -= 1
                    if nest == 0:
                        routine_end = j + 1
                        break
                j += 1
            if routine_end is None:
                print(f"Warning: Skipping {routine_type} {routine_name} (line {routine_start}) - no matching 'END' found.")
                while j < len(lines):
                    if header_pattern.match(lines[j]) or pkg_pattern.search(lines[j]) or re.search(r'^\s*end\s+[^;]*;', lines[j], re.IGNORECASE):
                        break
                    j += 1
                i = j
                continue
            routines.append({
                'type': routine_type,
                'name': routine_name,
                'start': routine_start,
                'end': routine_end,
                'schema': current_schema,
                'package': current_package
            })
            i = j
        else:
            i += 1

    return routines, lines

def extract_and_save_routines(input_file, output_dir):
    routines, lines = parse_routines(input_file)
    if not routines:
        print("No PL/SQL routines found.")
        return

    for routine in routines:
        # Convert schema, package, and routine names to uppercase
        schema = (routine.get('schema') or "DEFAULT_SCHEMA").upper()
        package = (routine.get('package') or "DEFAULT_PACKAGE").upper()
        routine_name = routine['name'].upper()
        
        # Create folder structure with uppercase names
        output_folder = os.path.join(output_dir, schema, package)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        routine_type = routine['type']
        start = routine['start']
        end = routine['end']
        routine_lines = lines[start-1:end]
        
        # File name in uppercase
        output_filename = f"{routine_name}.SQL"  # Also uppercasing the extension
        output_path = os.path.join(output_folder, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as fout:
            fout.writelines(routine_lines)
        print(f"Saved {routine_type} {routine_name} (lines {start}-{end}) in {output_path}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_routines_to_files.py <input_file> <output_directory>")
        sys.exit(1)
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    extract_and_save_routines(input_file, output_dir)

if __name__ == '__main__':
    main()