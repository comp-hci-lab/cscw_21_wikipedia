import mwapi
import traceback
import numpy as np
import sys
import pdb
from itertools import islice
import pandas as pd
import random
import mwreverts.api
from mw.lib import title as mwtitle
import traceback
from multiprocessing import Pool
from models import Page, Quality, Revision, User, Citation, RevVandalism,\
RevRevert
from functools import partial
from revscoring.utilities.util import dump_observation, read_observations
from json.decoder import JSONDecodeError
from peewee import fn
import json
import re

datadir = '../datasets/'
roles = ['bureaucrat', 'sysop', 'autoconfirmed', 'user']
main_header = ['rev_id', 'rev_page', 'rev_comment',
                'rev_user', 'rev_user_text', 'rev_timestamp', 'rev_minor_edit', 'rev_len',
                'rev_sha1', 'quality', 'rev_title']
user_header = ['user_id', 'user_name', 'user_registration', 'user_groups']
talk_header = ['rev_title',  'pageid', 'rev_timestamp',
        'rev_user', 'quality']
text_header = ['old_id', 'old_text']
article_history_regex =\
    r'{{articlehistory.*?}}'

article_history_section_regexs =[
    r'action{0}=([a-z]+)',
    'action{0}date=([: a-z0-9\(\),]+)',
    'action{0}link=([a-zA-Z_\/ ]+)',
    'action{0}result=([a-z]+)',
    'action{0}oldid=([0-9]+)']

actions_map = {0: 'action', 1: 'date', 2: 'link', 3: 'result', 4: 'oldid'}
actions_template = {v: '' for k, v in actions_map.items()}

class_map = {'non-def': 0, 'class=stub': 1, 'class=start': 2, 'class=c': 3, 'class=b': 4,
        'class=a': 5, 'class=ga': 6, 'class=fa': 7}


ENDPOINT = 'https://en.wikipedia.org'
quals = ['class=FA', 'class=GA', 'class=B', 'class=C', 'class=Stub', 'class=Start']
quals = [qual.lower() for qual in quals]
classes = quals

def normalize_ts(ts):
    if not ts:
        return ''
    return pd.Timestamp(ts).strftime('%Y%m%d%H%M%S')

def get_mwapi_session():
    return mwapi.Session(ENDPOINT, user_agent='Revision extractor, UMICH,')

def get_revs_for_vandalism_labeling(lim):
    revsdone = [rev.rev_id for rev in RevVandalism.select(RevVandalism.rev_id)]
    revs = Revision.select().where((Revision.rev_id.not_in(revsdone))).order_by(fn.Rand()).limit(lim)
    return list(revs)


def get_revs_for_revert_labeling(lim):
    if lim == -1:
        lim = Revision.select().count()
    revsdone = [rev.rev_id for rev in RevRevert.select(RevRevert.rev_id)]
    revs = Revision.select(Revision.rev_id).where((Revision.rev_id.not_in(revsdone))).order_by(fn.Rand()).limit(lim)
    return list(revs)


def get_user_info(session, users):
    userinfo = []
    users = [mwtitle.normalize(user) for user in users]
    users_from_db = User.select().where(User.user_name.in_(users))
    users_found = []
    for user in users_from_db:
        userinfo.append([user.user_id, user.user_name, user.user_registration,
            user.user_groups])
        users_found.append(user.user_name)
    users_toquery = [user for user in users if user not in users_found]

    print('Querying {0} users'.format(len(users_toquery)))
    doc = session.get(
            action='query', list='users', ususers=users_toquery, usprop=['groups',
                'groupmemberships', 'registration', 'blockinfo']
            )
    users_toadd = []
    for userob in doc['query']['users']:
        if 'userid'  not in userob:
            continue
        user_details = [userob['userid'], mwtitle.normalize(userob['name']),
                normalize_ts(userob['registration']),','.join(userob['groups'])]
        users_toadd.append(tuple(user_details))
        userinfo.append(user_details)
    #print('Insert {0} users'.format(len(users_toadd)))
    #User.insert_many(users_toadd, fields = user_header).execute()
    return pd.DataFrame(userinfo, columns = user_header)


def get_revert_status(session, rev_id):
    is_reverted = 0
    try:
        _, reverted, reverted_to = mwreverts.api.check(session, rev_id, radius = 15, window = 48*60*60)
        if reverted:
            is_reverted = 1
    except (RuntimeError, KeyError, JSONDecodeError) as e:
        sys.stderr.write('#')
    return rev_id, is_reverted

def add_page_qual(labeling, text):
    quality = ''
    for qual in quals:
        if qual in text:
            #if qual == 'class=ga':
            #    pdb.set_trace()
            quality = qual
            break
    labeling['quality'] = quality
    #labeling['size'] = len(text)

def get_section_regex_matchings(text):
    rem = True
    idx = 1
    actions_temp = actions_template.copy()
    actions = []
    while rem:
        found = False
        for index, regex in enumerate(article_history_section_regexs):
            reg = regex.format(idx)
            match = re.search(reg, text)
            if match:
                found = True
                actions_temp[actions_map[index]] = match.group(1)
        if found:
            actions.append(actions_temp)
        else:
            rem = False
        idx += 1
    return actions

def add_page_history(logger, labeling, text):
    history = re.search(article_history_regex, text, re.DOTALL)
    if not history:
        return
    matched = history.group(0)
    matchings = get_section_regex_matchings(matched)
    labeling['history'] = matchings


def get_revs_without_type(logger, lim):
    if lim == -1:
        lim = Revision.select(Revision.rev_id).count()
    revs = Revision.select(Revision.rev_id).where(Revision.rev_done == 0).order_by(fn.Rand()).limit(lim)
    return list(revs)


def get_page_text(session, labeling, key = 'article_pid', pageid = True):
    # Could be a pageid or revid
    articleid = labeling[key]
    try:
        article = None
        revisions = None
        ns = -1
        if pageid:
            revisions = session.get(action = 'query', prop = 'revisions', pageids = [articleid],
                    rvprop = ['ids'], formatversion = 2)
        else: 
            revisions = session.get(action = 'query', prop = 'revisions', revids = [articleid],
                    rvprop = ['ids'], formatversion = 2, continuation=True)
        for result in revisions:
            if 'query' not in result:
                continue
            pages = result['query']['pages']
            for page in pages:
                if 'revisions' not in page:
                    continue
                for rev in page['revisions']:
                    ns = page['ns']

        if pageid:
            article = session.get(action = 'parse' ,pageid = articleid, prop =
                    ['wikitext', 'revid'], redirects = True)
        else:
            article = session.get(action = 'parse', oldid = articleid, prop =
                    ['wikitext', 'revid'], redirects = True)
        text = article['parse']['wikitext']['*']
        labeling['article_text'] = text
        labeling['page_title'] = article['parse']['title']
        labeling['pageid'] = article['parse']['pageid']
        labeling['revid'] = article['parse']['revid']
        labeling['ns'] = ns
    except:
        #logger.info("Error {}".format(title))
        print('Error {0}'.format(traceback.format_exc()))
    return labeling

def get_page_stats_single(session, labeling):
    if 'quality' in labeling:
        return labeling
    try:
        pageid = labeling['talk_pid']
        talk = session.get(action = 'parse', pageid = pageid, prop =
                'wikitext', redirects = True)
        talk_text = talk['parse']['wikitext']['*'].lower()
        add_page_qual(labeling, talk_text)
    except:
        #logger.info("Error {}".format(title))
        print('Error {0}'.format(traceback.format_exc()))
    return labeling
    

def get_page_stats(session, labelings):
    try:
        #logger.info('Attempting to create page: {0}'.format(redirected_title))
        #print('Attempting to create page: {0}'.format(redirected_title))
        #page, created = Page.get_or_create(page_id = pageid, page_title =
        #        redirected_title, page_namespace = 0)

        #if not created:
        #    raise Exception('Could not add page: {0}'.format(redirected_title))
        pageid_to_labeling = {labeling['talk_pid']: labeling for labeling in
                labelings}
        talkids = list(pageid_to_labeling.keys())
        talks = session.get(action='query', pageids=talkids, prop='revisions',
                redirects=True, rvprop = ['content'], formatversion = 2, rvslots
                = 'main', continuation = True)
        for result in talks:
            if 'query' not in result:
                continue
            pages = result['query']['pages']
            for page in pages:
                if 'revisions' not in page:
                    continue
                for rev in page['revisions']:
                    pageid = page['pageid']
                    content = rev.get('slots', {}).get('main',
                            {}).get('content', '').lower()
                    labeling = pageid_to_labeling.get(pageid, {})
                    add_page_qual(labeling, content)

        labelings = pageid_to_labeling.values()
    except:
        #logger.info("Error {}".format(title))
        print('Error {0}'.format(traceback.format_exc()))
    return list(labelings)


def get_qual_samples(labelings, samples, mid_level_path):
    mid_level = {}
    with open(mid_level_path) as f_mid:
        mid_level = json.loads(f_mid.read())
    filtered = {cat: [] for cat in classes}
    random.shuffle(labelings)
    #labelings = sorted(labelings, key = lambda k: k.get('size', 0), reverse=True)
    for ob in labelings:
        if 'quality' not in ob or ob['quality'] == '':
            continue
        if len(filtered[ob['quality']]) > samples:
            continue
        if 'quality' in ob and ob['quality'] in classes and 'history' in ob:
            filtered[ob['quality']].append(ob)
        if all(len(val) > samples for val in filtered.values()):
            break
    return filtered

def get_page_edits(session, labeling):
    get_quals = False
    pageid = labeling['page_id']
    page_revs_arr = []
    # First, try to look for the pageid in the database
    #page_present = Revision.select().where(Revision.rev_page ==
    #        pageid).count()
    #if page_present > 0:
    revs_content = []
    if False:
        page_revs = []
        # get results
        revs = (Page
                .select(Page, Revision)
                .join(Revision, on=(Page.page_id==Revision.rev_page).alias('rev'))
                .where(Page.page_id == pageid))
        for page in revs:
            rev = page.rev
            page_revs.append([rev.rev_id, rev.rev_page, rev.rev_comment,
                    rev.rev_user, rev.rev_user_text, rev.rev_timestamp,
                    rev.rev_minor_edit, rev.rev_len,
                    rev.rev_sha1, rev.rev_quality, page.page_title])

        return pd.DataFrame(page_revs, columns = main_header)
    elif True: #page_present == 0:
        page_revs = []
        doc = session.get(
            action='query', prop='revisions', formatversion=2,
            pageids = pageid, rvprop=['ids', 'timestamp', 'user', 'size', 'comment',
                'sha1', 'flags', 'userid'],
            continuation=True, rvlimit=50, rvslots='main', rvdir='newer')
        title = ''
        try:
            for result in doc:
                if 'query' not in result:
                    continue
                pages = result['query']['pages']
                for page in pages:
                    if 'revisions' not in page:
                        continue
                    for rev in page['revisions']:
                        revid = rev['revid']
                        pageid = page['pageid']
                        title = mwtitle.normalize(page['title'])
                        ts = pd.Timestamp(rev.get('timestamp', '')).strftime('%Y%m%d%H%M%S')
                        content = rev.get('slots', {}).get('main', {}).get('content', '')
                        revs_content.append([revid, content])
                        rev = (
                                revid,
                                pageid,
                                rev.get('comment', ''),
                                rev.get('userid'),
                                rev.get('user'),
                                ts,
                                1 if rev.get('minor', False) else 0,
                                rev.get('size', 0),
                                rev.get('sha1', '')
                                )

                        page_revs.append(rev)
        except (json.decoder.JSONDecodeError, ValueError):
            sys.stderr.write("#")
            sys.stderr.flush()
        talk_id = labeling['talk_page_id']
        if get_quals:
            talk_edits = get_page_qualities(session, talk_id)
            page_revs_with_quals = []
            for rev in page_revs:
                ts = rev[5]
                t_filter = talk_edits['rev_timestamp'] < ts
                rev_new = list(rev)
                rev_new[5] = pd.Timestamp(ts).strftime('%Y%m%d%H%M%S')
                if not np.all(talk_edits.where(t_filter)['rev_timestamp'].isna()):
                    try:
                        max_ts = talk_edits.where(t_filter)['rev_timestamp'].idxmax()
                        quals.append(talk_edits.loc[max_ts, 'quality'])
                        rev_new = list(rev)
                        rev_new[5] = pd.Timestamp(ts).strftime('%Y%m%d%H%M%S')
                        page_revs_with_quals.append(tuple(rev_new) + (class_map[quals[-1]],))
                    except:
                        pdb.set_trace()
                        sys.stderr.write('#')
                        sys.stderr.flush()
                        print(traceback.format_exc())
                        page_revs_with_quals.append(tuple(rev_new) + (0,))
                        quals.append('non-def')
                else:
                    sys.stderr.write('#')
                    sys.stderr.flush()
                    page_revs_with_quals.append(tuple(rev_new) + (0,))
                    quals.append('non-def')


    #pdb.set_trace()
    #Revision.insert_many(page_revs_with_quals,fields=['rev_id', 'rev_page',
    #    'rev_comment', 'rev_user', 'rev_user_text', 'rev_timestamp',
    #    'rev_minor_edit', 'rev_len', 'rev_sha1' ,'rev_quality']).execute()
    #Page.insert(page_id = pageid, page_namespace = 0, page_title =
    #        title).execute()
    #for rev in page_revs:
    #    page_revs_arr.append([rev.rev_id, rev.rev_page, title,
    #        rev.rev_comment, rev.rev_user, rev.rev_user_text, rev.rev_timestamp,
    #        rev.rev_minor_edit, rev.rev_len, rev.rev_sha1])
    if get_quals:
        edits = pd.DataFrame([rev+(title,) for rev in page_revs_with_quals], columns = main_header)
    else:
        edits = pd.DataFrame([rev+(0, title,) for rev in page_revs], columns = main_header)

    revs_content = pd.DataFrame(revs_content, columns = text_header)
    return edits, revs_content

def get_page_qualities(session, pageid):
    page_revs = []
    doc = session.get(
        action='query', prop='revisions', formatversion=2,
        pageids=pageid, rvprop=['content', 'timestamp', 'user'],
        continuation=True, rvlimit=50, rvslots='main', rvdir='newer')
    for result in doc:
        pages = result['query']['pages']
        if len(result['query']['pages']) == 0:
            continue
        page = result['query']['pages'][0]
        if 'revisions' not in page:
            continue
        for rev in page['revisions']:
            if 'slots' not in rev or 'main' not in rev['slots'] or 'content' not in rev['slots']['main']:
                continue
            if 'user' not in rev or 'timestamp' not in rev:
                continue

            content = rev['slots']['main']['content'].lower()
            for qclass in quals:
                if qclass in content:
                    #sents = re.findall(r'[^.]*?'+qclass+r'[^.]*?\.',
                    #        content)
                    norm_title = mwtitle.normalize(page['title'][5:])
                    page_revs.append([norm_title, str(page['pageid']),
                        pd.to_datetime(str(rev['timestamp'])), rev['user'], qclass])
    return pd.DataFrame(page_revs, columns = talk_header)


def get_page_revs_with_quals(session, labeling):
    quals = []
    pageid = labeling['page_id']
    talk_id = labeling['talk_page_id']
    edits = get_page_edits(session, labeling)
    return edits


def get_articles_with_titles(titles, fpath):
    data = read_observations(open(datadir + fpath))
    labelings = list(data)
    return [labeling for labeling in labelings if labeling['talk_page_title'] in titles]


def get_all_page_revs(session, labelings):
    pass


def filter_present_articles(observations):
    pageids = [ob['pageid'] for ob in observations]
    pages_present = [page.page_id for page in
            Page.select().where(Page.page_id.in_(pageids))]
    return [ob for ob in observations if ob['pageid'] not in pages_present]




def chunkify(iterable, size = 1):
    l = len(iterable)
    for ndx in range(0, l, size):
        yield iterable[ndx:min(ndx + size, l)]

def batch(iterable, size):
    while True:
        batch = list(islice(iterable, 0, size))
        if len(batch) > 0:
            yield batch
        else:
            break
