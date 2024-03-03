import os
import re

project_dir = '.'
consolidated_file = 'consolidated_project_context.py'

with open(consolidated_file, 'w') as outfile:
  for root, dirs, files in os.walk(project_dir):
    for file in files:
      if file.endswith('.py'):
        filepath = os.path.join(root, file)
        relpath = os.path.relpath(filepath, project_dir)

        # Write header with relative path
        outfile.write(f'#\n# {relpath}\n#\n')

        # Read and write content, removing any trailing newline
        with open(filepath) as infile:
          content = infile.read().rstrip('\n')  # Remove trailing newline
          outfile.write(content)
          outfile.write('\n\n')  # Add blank lines for separation
