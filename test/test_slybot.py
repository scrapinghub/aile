# -*- coding: utf-8 -*-
import os
import tempfile
import json
import shutil
import collections
import contextlib

import aile


try:
    FILE = __file__
except NameError:
    FILE = './test'

TESTDIR = os.getenv('DATAPATH',
                    os.path.dirname(os.path.realpath(FILE)))


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


class CheckExtractionCannotFindField(Exception):
    pass


class CheckExtractionDifferentNumberOfItems(Exception):
    def __init__(self, expected, found):
        self.expected = expected
        self.found = found

    def __str__(self):
        return 'Different number of items. Expected: {0}, Found: {1}'.format(
            self.expected, self.found)


class CheckExtractionCannotFindItem(Exception):
    def __init__(self, item):
        self.item = item

    def __str__(self):
        return "Couldn't extract: {0}".format(self.item)


def _check_extraction(items, true_items):
    fields = find_fields(items[0], true_items[0])
    if not all(fields):
        raise CheckExtractionCannotFindField()
    if len(items) != len(true_items):
        raise CheckExtractionDifferentNumberOfItems(len(true_items), len(items))
    for true in true_items:
        any_match = False
        for extracted in items:
            match = True
            for field, true_value in zip(fields, true):
                if extracted[field][0].strip() != true_value:
                    match = False
                    break
            if match:
                any_match = True
        if not any_match:
            raise CheckExtractionCannotFindItem(true)
    return True

def check_extraction(all_items, true_items):
    found = False
    for items in all_items:
        try:
            found = _check_extraction(items, true_items)
            if found:
                break
        except CheckExtractionCannotFindField:
            pass
    assert found


PATCH_OF_LAND_1 = [
    ['158 Halsey Street, Brooklyn, New York'],
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
    ['2357 Greenfield Ave, Los Angeles, California']
]


def test_patchofland_1():
    with contextlib.closing(ExtractTest('Patch of Land.html')) as test:
        check_extraction(
            test.run('Patch of Land.html'), PATCH_OF_LAND_1)

MONSTER_1 = [
    [u'Java Developer (Graduate) Trading'],
    [u'C# Developer / C# .Net Programmer - R&D'],
    [u'Cyber Security Analyst - SIEM, CISSP, Vulnerability'],
    [u'UKEN Std Apply Auto Test Job With no RF and Q - Do not Apply'],
    [u'UKEN Std & Exp Apply Auto Test Job Without Questions - Do not Apply'],
    [u'UKEN Std Apply Auto Test Job Without Questions - Do not Apply'],
    [u'UKEN Std & Exp Apply Auto Test Job With Questions - Do not Apply'],
    [u'UKEN Express Apply Automation Test Job without Questions - Do not Apply'],
    [u'UKEN Company Confidential Test Job With Questions - Do not Apply'],
    [u'UKEN Std Apply Auto Test Job With Questions - Do not Apply'],
    [u'UKEN Shared Std Apply Auto Test Job - Do not Apply'],
    [u'UKEN Shared Apply Automation Test Job - Do not Apply'],
    [u'Front-End Developer , £30-35K'],
    [u'Ruby Developer - Fastest Growing Healthcare Startup'],
    [u'C#, MVC, Sharepoint, ASP.NET System Analyst/Developers'],
    [u'Oracle Applications DBA'],
    [u'C++ Developer Contract'],
    [u'Network Security Engineer -SolarWinds Specialist'],
    [u'Software Development Manager/CTO - Unified Communications'],
    [u'Senior C++ EFX Developer']
]


def test_monster_1():
    with contextlib.closing(ExtractTest('Monster.html')) as test:
        check_extraction(
            test.run('Monster.html'), MONSTER_1)


ARS_TECHNICA_1 = [
    [u'“Unauthorized code” in Juniper firewalls decrypts encrypted VPN traffic'],
    [u'LifeLock ID protection service to pay record $100 million for failing customers'],
    [u'Hacker hacks off Tesla with claims of self-driving car'],
    [u'Windows 10 Mobile upgrade won’t hit older phones until 2016'],
    [u'Video memories, storytelling, and Star Wars spoilers (no actual spoilers!)'],
    [u'Blackberry CEO says Apple has gone to a “dark place” with pro-privacy stance'],
    [u'Google ramps up EU lobbying as antitrust charges proceed'],
    [u'Dealmaster: Get a 32GB Moto X Pure Edition unlocked smartphone for $349'],
    [u'Apple gets a new COO, puts Phil Schiller in charge of the App Store'],
    [u'Microsoft makes 16 more Xbox 360 games playable on Xbox One'],
    [u'League of Legends now owned entirely by Chinese giant Tencent'],
    [u'Busted by online package tracking, drug dealer gets more than 8 years in prison'],
    [u'OneDrive for Business to get unlimited storage for enterprise customers'],
    [u'Germany approves 30-minute software update fix for cheating Volkswagen diesels'],
    [u'''Turing’s Shkreli on drug price-hike: “It gets people talking… that’s what art is”'''],
    [u'Self-driving Ford Fusions are coming to California next year'],
    [u'Cop who wanted to photograph teen’s erection in sexting case commits suicide'],
    [u'Republicans in Congress let net neutrality rules live on (for now)'],
    [u'Confirmed: Kojima leaves Konami to work on PS4 console exclusive [Updated]'],
    [u'Netflix to offer less bandwidth for My Little Pony , more for Avengers'],
    [u'Final NASA budget bill fully funds commercial crew and Earth science'],
    [u'Firefox for Windows finally has an official, stable 64-bit build'],
    [u'Smash Bros. DLC concludes with Bayonetta, Super Mario RPG Geno costume'],
    [u'New XPRIZE competition looks for a better underwater robot'],
    [u'Google’s new data-only Project Fi tablet plans don’t charge device fees'],
    [u'Tech firms could owe up to 4% of global revenue if they violate new EU data law'],
    [u'Android Pay adds in-app purchasing feature, catches up to Apple Pay'],
    [u'Pebble’s new Health app integrates with Timeline, suggests tips to get healthier'],
    [u'Websites may soon know if you’re mad—a little mouse will tell them'],
    [u'Dust Bowl returns as an Expedition in Oath of the Gatewatch'],
    [u'Orbitar, really? Some new exoplanet names are downright weird'],
    [u'13 million MacKeeper users exposed after MongoDB door was left open'],
]


def test_ars_technica_1():
    with contextlib.closing(ExtractTest('Ars Technica.html')) as test:
        check_extraction(
            test.run('Ars Technica.html'), ARS_TECHNICA_1)
