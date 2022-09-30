import json
import pathlib
import re

with open('settings.json', 'r') as f:
    settings = json.load(f)

input_filepath = pathlib.Path(settings['input_filepath'])

with open(input_filepath.as_posix(), 'r') as f:
    file_content = f.read()

if not settings['case_sensitive']:
    file_content = file_content.lower()

if settings['ignore_non_alphanumeric']:
    file_content = re.sub('[^a-zA-Z0-9]', '', file_content)

output_dict = dict()
for symbol in file_content:
    if symbol in output_dict:
        output_dict[symbol] += 1
    else:
        output_dict[symbol] = 1


if settings['sort_by'] == 'symbol':
    symbol_order = sorted(output_dict.keys(), reverse=(settings['order']=='desc'))
else:
    count_order = sorted(list(set(output_dict.values())), reverse=(settings['order']=='desc'))
    symbol_order = []
    for count in count_order:
        symbols = [key for key in output_dict if output_dict[key]==count]
        symbol_order.extend(symbols)

output_filepath = pathlib.Path(settings['output_filepath'])

with open(output_filepath.as_posix(), 'w') as f:
    for key in symbol_order:
        f.writelines(f'Symbol {key} occurs {output_dict[key]} times\n')