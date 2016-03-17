import os
import hashlib
import collections
import json

import pulp
import scrapely as sy
import slyd.utils
from slybot.plugins.scrapely_annotations.builder import Annotations

def tags_between(ptree, root, node):
    """Compute the tags that go from node upwards to root"""
    tags = []
    for node in ptree.prefix(node, stop_at=root):
        fragment = ptree.page.parsed_body[ptree.index[node]]
        if isinstance(fragment, sy.htmlpage.HtmlTag):
            tags.append(fragment.tag)
    return tags


class ItemLocation(tuple):
    """An item is a forest inside a PageTree. An item location is therefore
    a tuple where each element is the root of a tree"""
    pass


class FieldLocation(object):
    """Encodes the position of a field inside the PageTree

    Attributes
    ----------
    node : int
        Node number inside a PageTree
    item : ItemLocation
    root : int
        Root number inside item location
    sibling : int
    """
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
    """ Guess ftype value from the given field locations

    Parameters
    ----------
    ptree : PageTree
    locations : List[FieldLocation]

    Returns
    -------
    string
       One of the values: image, url, text
    """
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
    """Field schema and location inside the PageTree

    Attributes
    ----------
    name : string
        Field name
    required : bool
    vary : bool
    ftype : string
        Field type
    locations : List[FieldLocation]
    """
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


def item_location_tags(ptree, location):
    """Return a tags at item location

    Parameters
    ----------
    ptree : PageTree
    location : ItemLocation

    Returns
    -------
    List[HtmlTag]
    """
    last = location[-1]
    end = max(last + 1, ptree.match[last])
    if end >= len(ptree.index):
        end = ptree.index[-1]
    else:
        end = ptree.index[end]
    tags = ptree.page.parsed_body[ptree.index[location[0]]:end]
    return tags


def kmp_search(text, pattern):
    """
    Search pattern inside text using Knuth-Morris-Pratt algorithm.

    Implemented because it works on arbitrary sequences,
    like numpy arrays and lists, not just strings.

    Adaptated from:
    http://code.activestate.com/recipes/117214-knuth-morris-pratt-string-matching/
    """
    m = len(pattern)
    # build table of shift amounts
    shifts = [1] * (m + 1)
    shift = 1
    for pos in range(m):
        while shift <= pos and pattern[pos] != pattern[pos-shift]:
            shift += shifts[pos-shift]
        shifts[pos+1] = shift

    # do the actual search
    startPos = 0
    matchLen = 0
    for c in text:
        while matchLen == m or \
              matchLen >= 0 and pattern[matchLen] != c:
            startPos += shifts[matchLen]
            matchLen -= shifts[matchLen]
        matchLen += 1
        if matchLen == m:
            yield startPos


def common_prefix(sequences):
    """Given a collection of sequences, find the common prefix

    >>> common_prefix([[1,2,3],[1,2,3,4,5,6],[1,2]])
    [1, 2]
    """
    def all_equal(sequence):
        for element in sequence[1:]:
            if element != sequence[0]:
                return False
        return True
    common = []
    for x in zip(*sequences):
        if all_equal(x):
            common.append(x[0])
    return common


def common_suffix(sequences):
    """Given a collection of sequences, find the common suffix

    >>> common_suffix([[1,2,3,4,5], [0,3,4,5], [2,3,4,5]])
    [3, 4, 5]
    """
    return list(reversed(common_prefix(map(reversed, sequences))))


def suffix_jump(sequence, suffix):
    """How much to jump in sequence to skip the first occurence of suffix
    if there are more than one"""
    n = 0
    for pos in kmp_search(sequence, suffix):
        n += 1
        if n > 1:
            return prev_pos
        prev_pos = pos
    return 0


class Item(object):
    """ A collection of fields and the location of the item instances

    Attributes
    ----------
    name : string
        The item name
    ptree : PageTree
    locations : List[ItemLocation]
    fields : List[Field]

    common_prefix: List[HtmlFragment]
        Common prefix shared by all item locations
    common_suffix: List[HtmlFragment]
        Common suffix shared by all item locations
    min_jmp : int
        Parameter for scrapely tuning
    """
    def __init__(self, name, ptree, locations, fields):
        self.name = name
        self.ptree = ptree
        self.locations = locations
        self.fields = fields

        self.common_prefix = self._common_prefix()
        self.common_suffix = self._common_suffix()
        self.min_jump = self._min_jump()

    @property
    def dict(self):
        return {'fields': {field.name: field.dict for field in self.fields}}

    def _common_prefix(self):
        """Given the locations of this item, find the common prefix
        of all instances"""
        return common_prefix(
            item_location_tags(self.ptree, location)
            for location in self.locations)

    def _common_suffix(self):
        """Given the locations of this item, find the common suffix
        of all instances"""
        return common_prefix(
            item_location_tags(self.ptree, location)
            for location in self.locations)

    def _min_jump_location(self, location):
        return suffix_jump(
            item_location_tags(self.ptree, location), self.common_suffix)

    def _min_jump(self):
        return max(self._min_jump_location(location)
                   for location in self.locations)


def is_non_empty_text(page, fragment):
    """True if fragment contains non-empty text

    Parameters
    ----------
    page : HtmlPage
    fragment: int
        Fragment number inside parsed body
    """
    f = page.parsed_body[fragment]
    return f.is_text_content and page.body[f.start:f.end].strip()


def is_tag(page, fragment, tag=None):
    """True if fragment is HtmlTag

    Parameters
    ----------
    page : HtmlPage
    fragment : int
        Fragment number inside parsed body
    tag : Optional[string]
        Check if the tag is the same as this one
    """
    f = page.parsed_body[fragment]
    if isinstance(f, sy.htmlpage.HtmlTag):
        if tag is not None:
            return f.tag == tag
        else:
            return True
    return False


def is_link(page, fragment):
    """True if fragment is a link tag

    Parameters
    ----------
    page : HtmlPage
    fragment : int
        Fragment number inside parsed body
    """
    return is_tag(page, fragment, tag='a')


def is_image(page, fragment):
    """True if fragment is an image tag

    Parameters
    ----------
    page : HtmlPage
    fragment : int
        Fragment number inside parsed body
    """
    return is_tag(page, fragment, tag='img')


def default_is_of_interest(page, fragment):
    """Filter out non-interesting fragments

    This function is used as the default filter to find interesting fields
    for automatic extraction (extract_field_locations)
    """
    return (is_non_empty_text(page, fragment) or
            is_link(page, fragment) or
            is_image(page, fragment))


def extract_field_locations(ptree, item_location,
                            is_of_interest=default_is_of_interest):
    """Extract the locations of fields contained in item.

    Parameters
    ----------
    ptree : PageTree
    item_location : ItemLocation
    is_of_interest : func
        Only extract fields which pass this test
        Signature: (HtmlPage, int) -> bool

    Returns
    -------
    List[FieldLocation]
    """
    field_locations = set()
    for i, root in enumerate(item_location):
        for node in range(root, max(root + 1, ptree.match[root])):
            fragment = ptree.index[node]
            if is_of_interest(ptree.page, fragment):
                field_locations.add(
                    FieldLocation(node, item_location, i, ptree.i_child[node]))
    return field_locations


def group_fields_by_root(field_locations):
    """Given a list of field_locations group together those that hang from the
    same parent.

    Parameters
    ----------
    field_locations: List[FieldLocation]

    Returns
    -------
    dict
        Dictionary mapping PageTree node numbers to a list of FieldLocation
    """
    groups = collections.defaultdict(list)
    for field_location in field_locations:
        groups[field_location.item[field_location.root]].append(field_location)
    return groups


def group_fields_by_path(ptree, field_locations):
    """Two fields are considered equal if the path that goes from the field
    up to the item root is equal (and they have the same root index)
    """
    groups = collections.defaultdict(list)
    for field_location in field_locations:
        path_to_root = tags_between(
            ptree, field_location.item[field_location.root], field_location.node)
        groups[tuple(path_to_root)].append(field_location)
    return groups


def append_order(grouped_by_path):
    """Since two fields can have the same path we append an integer to
    the path to distinguish between them"""
    for path, field_locations in grouped_by_path.iteritems():
        for i, field_location in enumerate(field_locations):
            yield (path + (i,), field_location)


def group_fields(ptree, field_locations):
    """Group together equal field locations

    Parameters
    ----------
    ptree : PageTree
    field_locations : List[FieldLocation]

    Returns
    -------
    dict
        Dictionary mapping ordered paths to list of FieldLocation

    Example
    -------
    Equal field locations are those that have the same path from their
    item root to their location and, in case of several fields with the same path,
    at the same position. For example, consider this:

    0  <div name="Dell laptop"
    1     <img src="dell-laptop.png">
    2     <ul>
    3        <li name=price>
    4             600
    5        </li>
    6        <li name=ram>
    7             4GB
    8        </li>
    9     </ul>
    10 </div>
    11 <div name="Thinkpad laptop">
    12    <img src="thinkpad-laptop.png">
    13    <ul>
    14       <li name=price>
    15            1000
    16       </li>
    17       <li name=ram>
    18            8GB
    19       </li>
    20    </ul>
    21 </div>

    There are two items. The (ordered paths) to the fields would be:

    ordered_path      field_location
                      node    item
    (div, img)        1       0
    (div, ul, li, 0)  4       0
    (div, ul, li, 1)  7       0
    (div, img)        12      11
    (div, ul, li, 0)  15      11
    (div, ul, li, 1)  18      11


    And when grouped, the result would be a dict like this:

    {
       ('div', 'img'): [FieldLocation(1, (0,)), FieldLocation(12, (11,))],
       ('div', 'ul', 'li', 0): [FieldLocation(4, (0,)), FieldLocation(15, (11,))],
       ('div', 'ul', 'li', 1): [FieldLocation(7, (0,)), FieldLocation(18, (11,))]
    }
    """
    grouped = collections.defaultdict(list)
    by_root = group_fields_by_root(field_locations)
    for root, field_locations in by_root.iteritems():
        by_path = group_fields_by_path(ptree, field_locations)
        for path, field_location in append_order(by_path):
            grouped[path].append(field_location)
    return grouped


def cmp_location_groups(a, b):
    """Given two lists of FieldLocation, put first the one which has a
    location that appears first in the html page"""
    def get_nodes(x):
        return [f.node for f in x]
    return cmp(min(get_nodes(a)), min(get_nodes(b)))


def extract_fields(ptree, item_locations,
                   is_of_interest=default_is_of_interest, name='aile-field'):
    """Extract fields from page

    Parameters
    ----------
    ptree : PageTree
    item_locations: List[ItemLocation]
    is_of_interest: func
        With signature (HtmlPage, int) -> bool
    name : string
        Prefix for all fields name

    Returns
    -------
    """
    field_locations = [field_location
                       for item_location in item_locations
                       for field_location in extract_field_locations(
                               ptree, item_location, is_of_interest)]
    grouped = group_fields(ptree, field_locations)
    grouped_locations = sorted(grouped.values(), cmp=cmp_location_groups)
    return [Field(name='{0}-field-{1}'.format(name, i),
                  locations=locations,
                  required=False,
                  ftype=detect_field_type(ptree, locations))
            for i, locations in enumerate(grouped_locations)]


def extract_item(ptree, item_locations,
                 name='aile-item', is_of_interest=default_is_of_interest):
    """Extract items from page

    Parameters
    ----------
    ptree : PageTree
    item_locations: List[ItemLocation]
    is_of_interest: func
        With signature (HtmlPage, int) -> bool
    name : string
        Item name and prefix for all fields names

    Returns
    -------
    Item
    """
    return Item(name, ptree, item_locations,
                extract_fields(ptree, item_locations, is_of_interest, name=name))


def good_annotation_locations(item, annotate_first=True):
    """Find the minimum number of annotations necessary to extract all the fields

    Since annotations can be reviewed and modified later by the user we want to keep
    just the minimum number of them.

    Parameters
    ----------
    item : Item
    annotate_first : book
        If true always annotate the first instance of the item in the page

    Returns
    -------
    List[ItemLocation]
    """
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


def generate_item_annotations(item, best_locations=True,
                              container_name_prefix='aile-container'):
    """Generate annotations for the item

    Parameters
    ----------
    item : Item
    best_locations : bool
        If True try to reduce the annotations to a minimum number
    container-name: string
        Prefix to use for containers names

    Returns
    -------
    List[dict]
        Each dict represents an slybot annotation
    """
    def get_tagid(node):
        fragment = item.ptree.page.parsed_body[item.ptree.index[node]]
        if not isinstance(fragment, sy.htmlpage.HtmlTag):
            return get_tagid(item.ptree.parents[node])
        return int(fragment.attributes[slyd.utils.TAGID])

    container_node = item.ptree.common_ascendant(
        location[0] for location in item.locations)
    if container_node == -1:
        return
    container_name = '{0}-{1}'.format(container_name_prefix, item.name)
    yield {
        'annotations': {'content': '#listitem'},
        'id': container_name,
        'tagid': get_tagid(container_node),
        'item_container': True,
        'ptree_node': container_node
    }

    item_name = '{0}-first-instance'.format(item.name)
    location = item.locations[0]
    annotation = {
        'annotations': {'content': '#listitem'},
        'id': item_name,
        'tagid': get_tagid(location[0]),
        'item_container': True,
        'container_id': container_name,
        'item_id': item.name,
        'repeated': True,
        'ptree_node': location[0],
        'min_jump': item.min_jump,
        'max_separator': 0
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
    annotated = set()
    for i in annotation_locations:
        for j, (field_location, field_name) in enumerate(fields_in_location[item.locations[i]]):
            if field_name in annotated:
                continue
            annotated.add(field_name)
            fragment = item.ptree.index[field_location.node]
            if is_link(item.ptree.page, fragment):
                annotate = 'href'
            elif is_image(item.ptree.page, fragment):
                annotate = 'src'
            else:
                annotate = 'content'
            tagid = get_tagid(field_location.node)
            yield {
                'annotations': {annotate: field_name},
                'data': {
                    '{0}-{1}'.format(field_name, tagid): {
                        'attribute': annotate,
                        'field': field_name,
                        'extractors': [],
                        'required': False
                    }
                },
                'id': '{0}-instance-{1}'.format(field_name, i),
                'tagid': tagid,
                'item_container': False,
                'container_id': item_name,
                'schema_id': item.name,
                'ptree_node': field_location.node
            }


def merge_tagid_annotations(annotations):
    """If there are several annotations on the same tagid, merge them"""
    group_by_tagid = collections.defaultdict(list)
    for annotation in annotations:
        group_by_tagid[annotation['tagid']].append(annotation)
    for tagid, annotations_on_tagid in group_by_tagid.iteritems():
        first_annotation = annotations_on_tagid[0]
        if len(annotations_on_tagid) > 1:
            for other_annotation in annotations_on_tagid[1:]:
                first_annotation['annotations'].update(other_annotation['annotations'])
                first_annotation['data'].update(other_annotation['data'])
        yield first_annotation


def generate_project(name='AILE', version='1.0', comment=''):
    """Generate an Slybot project file"""
    return {
        'name': name,
        'version': version,
        'comment': comment
    }


def generate_spider(start_url, template_names):
    """Generate an slybot spider"""
    return {
        'start_urls': [start_url],
        'links_to_follow': 'none',
        'follow_patterns': [],
        'exclude_patterns': [],
        'template_names': template_names
    }


def generate_empty_template(page):
    """Generate an empty template(sample) page"""
    return {
        'extractors': {},
        'url': page.url,
        'original_body': page.body,
        'scrapes': 'aile-item-0',
        'page_type': 'item',
        'page_id': hashlib.md5(page.url).hexdigest(),
        "version": '0.13.0'
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


def merge_containers(annotations):
    """If there are several differemt items with the same container, reuse
    the container for all of them"""
    final_annotations = []
    grouped_by_tagid = collections.defaultdict(list)
    for annotation in annotations:
        if annotation.get('item_container'):
            grouped_by_tagid[annotation['tagid']].append(annotation)
        else:
            final_annotations.append(annotation)
    rename_id = {}
    for tagid, annotations in grouped_by_tagid.iteritems():
        if len(annotations) > 1:
            old_id = [annotation['id'] for annotation in annotations]
            new_id = '|'.join(old_id)
            new_annotation = annotations[0]
            new_annotation['id'] = new_id
            final_annotations.append(new_annotation)
            for annotation_id in old_id:
                rename_id[annotation_id] = new_id
        else:
            final_annotations.append(annotations[0])
    for annotation in final_annotations:
        container_id = annotation.get('container_id')
        if container_id in rename_id:
            annotation['container_id'] = rename_id[container_id]
    return sorted(final_annotations,
                  key=lambda annotation: annotation.get('item_id', ''))


def generate_from_samples(
        page_items,
        path='./slybot-project',
        spider_name='aile',
        min_item_fields=2,
        max_item_fields=None,):
    """Generate a full slybot project

    Parameters
    ----------
    page_items: List[(page, items)]
         page is an HtmlPage where tagids attributes have been added
         items is List[Item]
    path : string
        Directory where to store the project
    min_item_fields: int or None
        Discard items with less fields than this number
    max_item_fields: int or None
        Discard items with more fields than this number

    Returns
    -------
    None
    """
    if not os.path.exists(path):
        os.mkdir(path)

    # project.json
    with open(os.path.join(path, 'project.json'), 'w') as project_file:
        json.dump(generate_project(), project_file, indent=4, sort_keys=True)

    # project.json
    with open(os.path.join(path, 'project.json'), 'w') as project_file:
        json.dump(generate_project(), project_file, indent=4, sort_keys=True)

    # items.json
    all_items = collections.defaultdict(dict)
    for _, items in page_items:
        for item in items:
            for field_name, field_dict in item.dict['fields'].iteritems():
                all_items[item.name][field_name] = field_dict
    with open(os.path.join(path, 'items.json'), 'w') as items_file:
        json.dump({item_name: {'fields': fields}
                   for item_name, fields in all_items.iteritems()},
                  items_file, indent=4, sort_keys=True)

    # extractors
    with open(os.path.join(path, 'extractors.json'), 'w') as extractors_file:
        json.dump({}, extractors_file, indent=4, sort_keys=True)

    # spiders/
    spiders_dir = os.path.join(path, 'spiders')
    if not os.path.exists(spiders_dir):
        os.mkdir(spiders_dir)
    spider_dir = os.path.join(spiders_dir, spider_name)
    if not os.path.exists(spider_dir):
        os.mkdir(spider_dir)
    templates = []
    for i, (page, items) in enumerate(page_items):
        template = generate_empty_template(page)
        annotations = []
        for j, item in enumerate(filter(item_is_tag, items)):
            if min_item_fields is not None and len(item.fields) < min_item_fields:
                continue
            if max_item_fields is not None and len(item.fields) > max_item_fields:
                continue
            annotations += merge_tagid_annotations(generate_item_annotations(item))

        annotations = merge_containers(annotations)
        template['plugins'] = {
            'annotations-plugin': {'extracts': annotations }
        }
        Annotations().save_extraction_data({'extracts': annotations}, template)
        template_name = 'template-{0}'.format(i)
        template['name'] = template['id'] = template_name
        template_path = os.path.join(spider_dir, '{0}.json'.format(template_name))
        with open(template_path, 'w') as template_file:
            json.dump(template, template_file, indent=4, sort_keys=True)
        html_path = os.path.join(spider_dir, template_name + '-annotated.html')
        with open(html_path, 'w') as template_annotated:
            template_annotated.write(template['annotated_body'].encode('utf-8'))
        templates.append(template_name)

    spider_path = os.path.join(spiders_dir, '{0}.json'.format(spider_name))
    with open(spider_path, 'w') as spider_file:
        json.dump(
            generate_spider(page.url, templates),
            spider_file,
            indent=4,
            sort_keys=True)


def generate(item_extract, path='./slybot-project',
                    min_item_fields=2, max_item_fields=None, max_n_items=2):
    """Generate a full slybot project

    Warning: modifies item_extract.page_tree.page

    Parameters
    ----------
    item_extract : aile.kernel.ItemExtract
    items : List[Item]
    path : string
        Directory where to store the project
    min_item_fields: int or None
        Discard items with less fields than this number
    max_item_fields: int or None
        Discard items with more fields than this number
    max_n_items: int
        Maximum number of different item types. -1 to set no limit.

    Returns
    -------
    None
    """
    slyd.utils.add_tagids(item_extract.page_tree.page)
    items = [
        extract_item(
            item_extract.page_tree, item_locations, name='aile-item-{0}'.format(i))
        for i, item_locations in enumerate(item_extract.items[:max_n_items])
    ]
    generate_from_samples(
        page_items=[(item_extract.page_tree.page, items)],
        path=path,
        spider_name='aile',
        min_item_fields=min_item_fields,
        max_item_fields=max_item_fields)
