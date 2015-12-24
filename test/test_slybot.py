import os
import tempfile
import json
import shutil
import collections
import contextlib

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


class ExtractTest(object):
    def __init__(self, train_url):
        self.train_url = get_local_url(train_url)
        self.project_path = tempfile.mkdtemp()
        self.item_extract = aile.generate_slybot_project(
            self.train_url, path=self.project_path, verbose=False)
        with open(os.path.join(self.project_path, 'items.json'), 'r') as schema_file:
            schema = json.load(schema_file)
        self.schema = {
            item_name: set(item['fields'])
            for item_name, item in schema.iteritems()}

    def run(self, url=None):
        if url is None:
            url = self.train_url
        else:
            url = get_local_url(url)
        extract_path = tempfile.mktemp(suffix='.json')
        opt = [
            '-s LOG_LEVEL=CRITICAL',
            '-s SLYDUPEFILTER_ENABLED=0',
            '-s PROJECT_DIR={0}'.format(self.project_path),
            '-o {0}'.format(extract_path)
        ]
        cmd = 'slybot crawl {1} aile -a start_urls="{0}"'.format(url, ' '.join(opt))
        if os.system(cmd) != 0:
            return None
        with open(extract_path, 'r') as extract_file:
            items = json.load(extract_file)
        os.remove(extract_path)
        grouped_items = collections.defaultdict(list)
        for item in items:
            name = item_name(self.schema, item)
            if name:
                grouped_items[name].append(item)
        return grouped_items.values()

    def close(self):
        shutil.rmtree(self.project_path)


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
    with contextlib.closing(ExtractTest('Patch of Land.html')) as test:
        check_extraction(
            test.run('Patch of Land 2.html'),
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
