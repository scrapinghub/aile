import os
import tempfile
import json
import shutil
import collections

import aile


TESTDIR = os.getenv('DATAPATH',
                    os.path.dirname(os.path.realpath(__file__)))


def get_local_url(filename):
    return 'file:///{0}/{1}'.format(TESTDIR, filename)


def item_name(schema, item):
    for field in item.keys():
        for name, fields in schema.iteritems():
            if field in fields:
                return name
    return None


def extract_all_items(url):
    project_path = tempfile.mkdtemp()
    extract_path = tempfile.mktemp(suffix='.json')
    aile.generate_slybot_project(
        get_local_url(url), path=project_path, verbose=False)
    with open(os.path.join(project_path, 'items.json'), 'r') as schema_file:
        schema = json.load(schema_file)
    schema = {
        item_name: set(item['fields'])
        for item_name, item in schema.iteritems()}
    opt = [
        '-s LOG_LEVEL=CRITICAL',
        '-s SLYDUPEFILTER_ENABLED=0',
        '-s PROJECT_DIR={0}'.format(project_path),
        '-o {0}'.format(extract_path)
    ]
    ret = os.system('slybot crawl aile ' + ' '.join(opt))
    if ret != 0:
        return None
    with open(extract_path, 'r') as extract_file:
        items = json.load(extract_file)
    shutil.rmtree(project_path)
    os.remove(extract_path)

    grouped_items = collections.defaultdict(list)
    for item in items:
        name = item_name(schema, item)
        if name:
            grouped_items[name].append(item)
    return grouped_items.values()



def find_fields(item, true_item):
    fields = []
    for true_field_value in true_item:
        found = False
        for field_name, field_value in item.iteritems():
            if true_field_value == field_value[0].strip():
                found = True
                fields.append(field_name)
        if not found:
            fields.append(None)
    return fields


def _check_extraction(items, true_items):
    fields = find_fields(items[0], true_items[0])
    if not all(fields):
        return False
    if len(items) != len(true_items):
        return False
    for extracted, true in zip(items, true_items):
        for field, true_value in zip(fields, true):
            if extracted[field][0].strip() != true_value:
                return False
    return True


def check_extraction(all_items, true_items):
    assert any(_check_extraction(items, true_items)
               for items in all_items)


def test_patchofland():
    check_extraction(
        extract_all_items('Patch of Land.html'),
        [['158 Halsey Street, Brooklyn, New York'],
         ['695 Monroe Street, Brooklyn, New York'],
         ['138 Wood Road, Los Gatos, California'],
         ['Multiple Addresses, Sacramento, California'],
         ['438 29th St, San Francisco, California'],
         ['747 Kingston Road, Princeton, New Jersey'],
         ['2459 Ketchum Rd, Memphis, Tennessee'],
         ['158 Halsey Street, Brooklyn, New York'],
         ['697 Monroe St., Brooklyn, New York'],
         ['2357 Greenfield Ave, Los Angeles, California'],
         ['5567 Colwell Road, Penryn, California'],
         ['2357 Greenfield Ave, Los Angeles, California']]
    )


if __name__=='__main__':
    test_patchofland()
