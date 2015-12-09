import os
import hashlib
import collections
import json

import scrapely as sy
import pulp
from slyd.plugins.scrapely_annotations import Annotations


def tags_between(ptree, root, node):
    """Compute the tags that go from node upwards to root"""
    tags = []
    for node in ptree.prefix(node, stop_at=root):
        fragment = ptree.page.parsed_body[ptree.index[node]]
        if isinstance(fragment, sy.htmlpage.HtmlTag):
            tags.append(fragment.tag)
    return tags


class FieldLocation(object):
    def __init__(self, node, item, root=None, sibling=None):
        self.node = node
        self.item = item
        self.root = root if root is not None else 0
        # TODO
        self.sibling = None


class Field(object):
    def __init__(self, name, locations,
                 required=False, vary=False, ftype='text'):
        self.name = name
        self.required = required
        self.vary = vary
        self.ftype = ftype
        self.locations = locations

    @property
    def dict(self):
        return {
            'type': self.ftype,
            'required': self.required,
            'vary': self.vary,
        }


class Item(object):
    def __init__(self, name, ptree, locations, fields):
        self.name = name
        self.ptree = ptree
        self.locations = locations
        self.fields = fields

    @property
    def dict(self):
        return {'fields': {field.name: field.dict for field in self.fields}}


def extract_field_locations(ptree, item_location, is_of_interest=None):
    """Extract the locations of fields contained in item.

    Returns a list of FieldLocation
    """
    if is_of_interest is None:
        is_of_interest = lambda node: ptree.page.parsed_body[node].is_text_content
    field_locations = []
    for i, root in enumerate(item_location):
        for node in range(root, max(root + 1, ptree.match[root])):
                if is_of_interest(ptree.index[node]):
                    field_locations.append(FieldLocation(node, item_location, i))
    return field_locations


def group_fields_by_path(ptree, field_locations):
    """Two fields are considered equal if the path that goes from the field
    up to the item root is equal

    TODO: siblings
    """
    groups = collections.defaultdict(list)
    for field_location in field_locations:
        path_to_root = tags_between(
            ptree, field_location.item[field_location.root], field_location.node)
        groups[tuple(path_to_root)].append(field_location)
    return groups


def extract_fields(ptree, item_locations, is_of_interest=None, name='aile-field'):
    field_locations = [field_location
                       for item_location in item_locations
                       for field_location in extract_field_locations(ptree, item_location)]
    grouped = group_fields_by_path(ptree, field_locations)
    return [Field(name='{0}-field-{1}'.format(name, i),
                  locations=locations,
                  required=(len(item_locations)==len(locations)))
            for i, (path, locations) in enumerate(grouped.iteritems())]


def extract_item(ptree, item_locations, name='aile-item', is_of_interest=None):
    return Item(name, ptree, item_locations,
                extract_fields(ptree, item_locations, is_of_interest, name=name))


def good_annotation_locations(item):
    """Find the minimum number of annotations necessary to extract all the fields"""
    #    x[i] = 1 iff i-th item is representative
    # A[i, j] = 1 iff i-th item contains the j-th field
    #
    # Solve:
    #           min np.sum(x)
    # Subject to:
    #           np.all(np.dot(A.T, x) >= np.repeat(1, len(fields)))
    index_locations = {location: i for i, location in enumerate(item.locations)}
    P = pulp.LpProblem('good_annotation_locations', pulp.LpMinimize)
    X = [pulp.LpVariable('x{0}'.format(i), cat='Binary')
         for i in range(len(index_locations))]
    P += pulp.lpSum(X)
    for field in item.fields:
        P += pulp.lpSum(
            [X[index_locations[location.item]] for location in field.locations]) >= 1
    P.solve()
    return [i for (i, x) in enumerate(X) if x.value() == 1]


def generate_item_annotations(item):
    container_node = item.ptree.common_ascendant(
        location[0] for location in item.locations)
    yield {
        'annotations': {'content': '#listitem'},
        'id': 'aile-container',
        'required': [],
        'tagid': item.ptree.index[container_node],
        'item_container': True
    }

    fields_in_location = collections.defaultdict(list)
    for field in item.fields:
        for location in field.locations:
            fields_in_location[location.item].append(location)
    annotation_locations = good_annotation_locations(item)
    for i in annotation_locations:
        location = item.locations[i]
        item_id = 'aile-item-instance-{0}'.format(i)
        annotation = {
            'annotations': {'content': '#listitem'},
            'id': item_id,
            'required': [],
            'tagid': item.ptree.index[location[0]],
            'item_container': True
        }
        if len(location) > 1:
            annotation['siblings'] = len(location) - 1
        yield annotation
        for j, field_location in enumerate(fields_in_location[item.locations[i]]):
            yield {
                'annotations': {'content': 'text-content'},
                'id': '{0}-field-{1}'.format(item_id, j),
                'required': [],
                'tagid': item.ptree.index[field_location.node],
                'item_container': False,
                'container_id': item_id,
                'repeated_item': False,
                'item_id': item.name
            }


def generate_project(name='AILE', version='1.0', comment=''):
    return {
        'name': name,
        'version': version,
        'comment': comment
    }


def generate_spider(start_url, templates):
    return {
        'start_urls': [start_url],
        'links_to_follow': 'patterns',
        'follow_patterns': [],
        'exclude_patterns': [],
        'templates': templates
    }


def generate_empty_template(page, scrapes):
    return {
        'extractors': {},
        'annotated_body': '',
        'url': page.url,
        'original_body': page.body,
        'scrapes': scrapes,
        'page_type': 'item',
        'page_id': hashlib.md5(page.url).hexdigest()
    }


def generate_slybot(item_extract, path='./slybot-project'):
    if not os.path.exists(path):
        os.mkdir(path)

    # project.json
    with open(os.path.join(path, 'project.json'), 'w') as project_path:
        json.dump(generate_project(), project_path, indent=4, sort_keys=True)

    # items.json
    items = [
        extract_item(
            item_extract.page_tree, item_locations, name='aile-item-{0}'.format(i))
        for i, item_locations in enumerate(item_extract.items)
    ]
    items = [items[0]]
    with open(os.path.join(path, 'items.json'), 'w') as items_path:
        json.dump({item.name: item.dict for item in items},
                  items_path, indent=4, sort_keys=True)

    # extractors
    with open(os.path.join(path, 'extractors.json'), 'w') as extractors_path:
        json.dump({}, extractors_path, indent=4, sort_keys=True)

    # spiders/
    templates = []
    for item in items:
        annotations = generate_item_annotations(item)
        template = generate_empty_template(item_extract.page_tree.page, item.name)
        Annotations().save_extraction_data({'extracts': annotations}, template)
        templates.append(template)
    spiders_dir = os.path.join(path, 'spiders')
    if not os.path.exists(spiders_dir):
        os.mkdir(spiders_dir)
    with open(os.path.join(spiders_dir, 'aile.json'), 'w') as spider_path:
        json.dump(
            generate_spider(
                item_extract.page_tree.page.url,
                templates),
            spider_path,
            indent=4,
            sort_keys=True)
