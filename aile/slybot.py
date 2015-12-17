import os
import hashlib
import collections
import json

import scrapely as sy
import slyd.utils
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
        self.sibling = sibling

    def __hash__(self):
        return hash(self.node)

    def __repr__(self):
        return repr((self.node, self.item, self.root, self.sibling))

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return (self.node == other.node and
                self.item == other.item and
                self.root == other.root and
                self.sibling == other.sibling)


def detect_field_type(ptree, locations):
    types = []
    for location in locations:
        fragment = ptree.index[location.node]
        if is_image(ptree.page, fragment):
            types.append('img')
        elif is_link(ptree.page, fragment):
            types.append('a')
        else:
            types.append('text')
    if all(map(lambda x: x == 'img', types)):
        return 'image'
    elif all(map(lambda x: x == 'a', types)):
        return 'url'
    else:
        return 'text'

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


def is_non_empty_text(page, fragment):
    f = page.parsed_body[fragment]
    return f.is_text_content and page.body[f.start:f.end].strip()


def is_tag(page, fragment, tag=None):
    f = page.parsed_body[fragment]
    if isinstance(f, sy.htmlpage.HtmlTag):
        if tag is not None:
            return f.tag == tag
        else:
            return True
    return False


def is_link(page, fragment):
    return is_tag(page, fragment, tag='a')


def is_image(page, fragment):
    return is_tag(page, fragment, tag='img')


def default_is_of_interest(page, fragment):
    return (is_non_empty_text(page, fragment) or
            is_link(page, fragment) or
            is_image(page, fragment))



def extract_field_locations(ptree, item_location, is_of_interest=default_is_of_interest):
    """Extract the locations of fields contained in item.

    Returns a list of FieldLocation
    """
    field_locations = set()
    for i, root in enumerate(item_location):
        for node in range(root, max(root + 1, ptree.match[root])):
            tagid = ptree.index[node]
            if is_of_interest(ptree.page, tagid):
                if not isinstance(ptree.page.parsed_body[tagid], sy.htmlpage.HtmlTag):
                    node = ptree.parents[node]
                if node != -1:
                    field_locations.add(
                        FieldLocation(node, item_location, i, ptree.i_child[node]))
    return field_locations


def group_fields_by_path(ptree, field_locations):
    """Two fields are considered equal if the path that goes from the field
    up to the item root is equal (and they have the same root index)
    """
    groups = collections.defaultdict(list)
    for field_location in field_locations:
        path_to_root = tags_between(
            ptree, field_location.item[field_location.root], field_location.node)
        path_to_root.append(field_location.sibling)
        path_to_root.append(field_location.root)
        groups[tuple(path_to_root)].append(field_location)
    return groups


def cmp_location_groups(a, b):
    def get_nodes(x):
        return [f.node for f in x]
    return cmp(min(get_nodes(a)), min(get_nodes(b)))


def extract_fields(ptree, item_locations, is_of_interest=default_is_of_interest, name='aile-field'):
    field_locations = [field_location
                       for item_location in item_locations
                       for field_location in extract_field_locations(
                               ptree, item_location, is_of_interest)]
    grouped = group_fields_by_path(ptree, field_locations)
    grouped_locations = sorted(grouped.values(), cmp=cmp_location_groups)
    return [Field(name='{0}-field-{1}'.format(name, i),
                  locations=locations,
                  required=False,
                  ftype=detect_field_type(ptree, locations))
            for i, locations in enumerate(grouped_locations)]


def extract_item(ptree, item_locations, name='aile-item', is_of_interest=default_is_of_interest):
    return Item(name, ptree, item_locations,
                extract_fields(ptree, item_locations, is_of_interest, name=name))


def good_annotation_locations(item, annotate_first=True):
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
    if annotate_first:
        P += X[0] == 1
    P += pulp.lpSum(X)
    for field in item.fields:
        P += pulp.lpSum(
            [X[index_locations[location.item]] for location in field.locations]) >= 1
    P.solve()
    return [i for (i, x) in enumerate(X) if x.value() == 1]


def generate_item_annotations(item, best_locations=True):
    def get_tagid(i):
        return item.ptree.page.parsed_body[i].attributes[slyd.utils.TAGID]

    container_node = item.ptree.common_ascendant(
        location[0] for location in item.locations)
    yield {
        'annotations': {'content': '#listitem'},
        'id': 'aile-container',
        'tagid': get_tagid(item.ptree.index[container_node]),
        'item_container': True,
        'ptree_node': container_node
    }

    location = item.locations[0]
    annotation = {
        'annotations': {'content': '#listitem'},
        'id': 'aile-item-first-instance',
        'tagid': get_tagid(item.ptree.index[location[0]]),
        'item_container': True,
        'container_id': 'aile-container',
        'item_id': item.name,
        'repeated': True,
        'ptree_node': location[0]
    }
    if len(location) > 1:
        annotation['siblings'] = len(location) - 1
    yield annotation
    fields_in_location = collections.defaultdict(list)
    for field in item.fields:
        for location in field.locations:
            fields_in_location[location.item].append((location, field.name))
    if best_locations:
        annotation_locations = good_annotation_locations(item)
    else:
        annotation_locations = [0]
    for i in annotation_locations:
        for j, (field_location, field_name) in enumerate(fields_in_location[item.locations[i]]):
            fragment = item.ptree.index[field_location.node]
            if is_link(item.ptree.page, fragment):
                annotate = 'href'
            elif is_image(item.ptree.page, fragment):
                annotate = 'src'
            else:
                annotate = 'content'
            yield {
                'annotations': {annotate: field_name},
                'id': '{0}-instance-{1}'.format(field_name, i),
                'tagid': get_tagid(fragment),
                'item_container': False,
                'container_id': 'aile-item-first-instance',
                'item_id': item.name,
                'ptree_node': field_location.node
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
        'links_to_follow': 'none',
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


def item_is_tag(item):
    """True if all the locations of the items are tag"""
    for location in item.locations:
        for root in location:
            if not isinstance(
                    item.ptree.page.parsed_body[item.ptree.index[root]],
                    sy.htmlpage.HtmlTag):
                return False
    return True


def generate_slybot(item_extract, path='./slybot-project', min_item_fields=2, max_item_fields=50):
    """Warning: modifies item_extract.page_tree.page"""
    slyd.utils.add_tagids(item_extract.page_tree.page)

    if not os.path.exists(path):
        os.mkdir(path)

    # project.json
    with open(os.path.join(path, 'project.json'), 'w') as project_file:
        json.dump(generate_project(), project_file, indent=4, sort_keys=True)

    # items.json
    items = [
        extract_item(
            item_extract.page_tree, item_locations, name='aile-item-{0}'.format(i))
        for i, item_locations in enumerate(item_extract.items)
    ]
    with open(os.path.join(path, 'items.json'), 'w') as items_file:
        json.dump({item.name: item.dict for item in items},
                  items_file, indent=4, sort_keys=True)

    # extractors
    with open(os.path.join(path, 'extractors.json'), 'w') as extractors_file:
        json.dump({}, extractors_file, indent=4, sort_keys=True)

    # spiders/
    templates = []
    for item in filter(item_is_tag, items):
        if min_item_fields is not None and len(item.fields) < min_item_fields:
            continue
        if max_item_fields is not None and len(item.fields) > max_item_fields:
            continue
        annotations = list(generate_item_annotations(item))
        with open(os.path.join(path, 'annotation-{0}.json'.format(item.name)), 'w') as annotation_file:
            json.dump(annotations, annotation_file, indent=4, sort_keys=True)
        template = generate_empty_template(item_extract.page_tree.page, item.name)
        Annotations().save_extraction_data({'extracts': annotations}, template)
        with open(os.path.join(path,
                               '{0}-template.html'.format(item.name)),
                  'w') as template_file:
            template_file.write(template['annotated_body'].encode('UTF-8'))
        templates.append(template)

    spiders_dir = os.path.join(path, 'spiders')
    if not os.path.exists(spiders_dir):
        os.mkdir(spiders_dir)
    with open(os.path.join(spiders_dir, 'aile.json'), 'w') as spider_file:
        json.dump(
            generate_spider(
                item_extract.page_tree.page.url,
                templates),
            spider_file,
            indent=4,
            sort_keys=True)
