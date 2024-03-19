import json, argparse
import pandas as pd
import dill as pickle
from tqdm import tqdm
from collections import Counter
from pathlib import Path


def parse_arguments():
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(description='Create CoT instruction data.')
    parser.add_argument('-dp', '--data_path', type=str, default='../../datasets', help="Path to raw datasets")
    parser.add_argument('-trp', '--trans_rsd_path', type=str, default='./translate/output', help="Path to the translated rsd files.")
    parser.add_argument('-o', '--output_dir', type=str, default='./output', help="Path to save the translated texts")
    return parser.parse_args()


def get_events(parent_uid, 
               child_uid, 
               claim_frames_df, 
               evt_kes_df, 
               evt_slots_df, 
               rel_kes_df, 
               rel_slots_df, 
               arg_kes_df, 
               dwd_overlay
              ):
    """
    Retrieve events for a document from the annotation data.
    Given a "parent_uid" of a document in parent_children.tab, find the "root_uid" (== "parent_uid") and 
    "associated_kes" in claim_frames.tab. Match them with the "root_uid" and "eventmention_id" in 
    evt_kes.tab and retrieve the "description" of the event. Then, determine the event type name and its definition 
    using the "qnode_type_id" and the DWD Overlay. After that, retrieve the "argmention_id" in evt_slots.tab 
    using "root_uid" and "eventmention_id" as queries. Use the "root_uid" and "argmention_id" as queries to find
    the event arguments by retrieving the "description" in arg_kes.tab. Also in arg_kes.tab, get the "qnode_type_id" 
    and use it to get the argument role in the DWD Overlay.
    """
    
    events = []
    root_uid = parent_uid
    associated_kes = claim_frames_df[claim_frames_df['root_uid']==root_uid]['associated_kes'].values
    for kes in associated_kes:
        for ke in kes.split(','):
            # ke can start with 'VM' (event mention id) or 'RM' (relation mention id).
            # Here, for event extraction, we only need those start with 'VM'.
            if ke.startswith('VM'):
                event = dict()
                
                # Retrieve the event description.
                event_mention = evt_kes_df[(evt_kes_df['root_uid']==root_uid) & 
                                           (evt_kes_df['eventmention_id']==ke)]['description'].values[0]
                event['event_mention'] = event_mention
                
                # Retrieve the qnode_type_ids.
                event_type_ids = evt_kes_df[(evt_kes_df['root_uid']==root_uid) & 
                                            (evt_kes_df['eventmention_id']==ke)]['qnode_type_id'].values[0]
                event_type_ids = event_type_ids.strip().split(' | ')
                
                event['event_types'] = []
                # Retrieve the event type name and definition from DWD Overlay.
                for event_type_id in event_type_ids:
                    dwd_qnode = 'DWD_' + event_type_id
                    try:
                        event_type_name = dwd_overlay['events'][dwd_qnode]['name']
                        event_type_definition = dwd_overlay['events'][dwd_qnode]['wd_description']
                    except:
                        event_type_name = "unknown"
                        event_type_definition = ""
                        
                    event_type = {'event_type_name': event_type_name, 'event_type_definition': event_type_definition}
                    event['event_types'].append(event_type)
                    
                    
                # Use "root_uid" and "eventmention_id" in evt_kes.tab to retrieve the "argmention_id" in evt_slots.tab.
                argmention_ids = evt_slots_df[(evt_slots_df['root_uid']==root_uid) & 
                                              (evt_slots_df['eventmention_id']==ke)]['argmention_id'].values
                
                event['arguments'] = []
                event['argument_roles'] = []
                
                # Retrieve the event argument descriptions and argument roles from DWD Overlay.
                for argmention_id in argmention_ids:
                    # The argument of an event can be an entity or another event or a relation.
                    # Thus, the argmention_id can start with 'EM' or 'VM' or 'RM'.
                    
                    # If the current event argument is an entity ('EM'), 
                    if argmention_id.startswith('EM'):
                        # Retrieve the event argument description.
                        argument_mention = arg_kes_df[(arg_kes_df['root_uid']==root_uid) & 
                                                      (arg_kes_df['argmention_id']==argmention_id)]['description'].values[0]
                        # Retrieve the qnode_type_id.
                        argument_role_ids = arg_kes_df[(arg_kes_df['root_uid']==root_uid) & 
                                                       (arg_kes_df['argmention_id']==argmention_id)]['qnode_type_id'].values[0]
                        argument_role_ids = argument_role_ids.strip().split(' | ')
                        
                        event_argument = {'argument_type': 'EM', 'argument_mention': argument_mention}
                        event['arguments'].append(event_argument)
                        
                        # Retrieve the argument role name and definition from DWD Overlay.
                        for argument_role_id in argument_role_ids:
                            dwd_qnode = 'DWD_' + argument_role_id
                            try:
                                argument_role_name = dwd_overlay['entities'][dwd_qnode]['name']
                                argument_role_definition = dwd_overlay['entities'][dwd_qnode]['wd_description']
                            except:
                                try:
                                    argument_role_name = dwd_overlay['events'][dwd_qnode]['name']
                                    argument_role_definition = dwd_overlay['events'][dwd_qnode]['wd_description']
                                except:
                                    try:
                                        argument_role_name = dwd_overlay['relations'][dwd_qnode]['name']
                                        argument_role_definition = dwd_overlay['relations'][dwd_qnode]['wd_description']
                                    except:
                                        argument_role_name = "unknown"
                                        argument_role_definition = ""
                                        
                            argument_role = {
                                'argument_type': 'EM', 
                                'argument_role_name': argument_role_name, 
                                'argument_role_definition': argument_role_definition
                            }
                            event['argument_roles'].append(argument_role)
                            
                    # If the current event argument is another event ('VM'),
                    elif argmention_id.startswith('VM'):
                        # Retrieve the event description (event mention) as the argument mention.
                        argument_mention = evt_kes_df[(evt_kes_df['root_uid']==root_uid) & 
                                                      (evt_kes_df['eventmention_id']==argmention_id)]['description'].values[0]
                        event_argument = {'argument_type': 'VM', 'argument_mention': argument_mention}
                        event['arguments'].append(event_argument)

                        # Retrieve the event type as the argument role.
                        # Retrieve the qnode_type_ids.
                        argument_role_ids = evt_kes_df[(evt_kes_df['root_uid']==root_uid) & (evt_kes_df['eventmention_id']==argmention_id)]['qnode_type_id'].values[0]
                        argument_role_ids = argument_role_ids.strip().split(' | ')

                        for argument_role_id in argument_role_ids:
                            dwd_qnode = 'DWD_' + argument_role_id
                            try:
                                argument_role_name = dwd_overlay['events'][dwd_qnode]['name']
                                argument_role_definition = dwd_overlay['events'][dwd_qnode]['wd_description']
                            except:
                                try:
                                    argument_role_name = dwd_overlay['entities'][dwd_qnode]['name']
                                    argument_role_definition = dwd_overlay['entities'][dwd_qnode]['wd_description']
                                except:
                                    try:
                                        argument_role_name = dwd_overlay['relations'][dwd_qnode]['name']
                                        argument_role_definition = dwd_overlay['relations'][dwd_qnode]['wd_description']
                                    except:
                                        argument_role_name = "unknown"
                                        argument_role_definition = ""

                            argument_role = {'argument_type': 'VM', 'argument_role_name': argument_role_name, 'argument_role_definition': argument_role_definition}
                            event['argument_roles'].append(argument_role)

                    # If the current event argument is a relation ('RM'),
                    elif argmention_id.startswith('RM'):
                        # Retrieve the relation description as the argument mention.
                        argument_mention = rel_kes_df[(rel_kes_df['root_uid']==root_uid) & (rel_kes_df['relationmention_id']==argmention_id)]['description'].values[0]
                        event_argument = {'argument_type': 'RM', 'argument_mention': argument_mention}
                        event['arguments'].append(event_argument)
                        
                        # Retrieve the relation type as the argument role.
                        # Retrieve the qnode_type_ids.
                        argument_role_ids = rel_kes_df[(rel_kes_df['root_uid']==root_uid) & (rel_kes_df['relationmention_id']==argmention_id)]['qnode_type_id'].values[0]
                        argument_role_ids = argument_role_ids.strip().split(' | ')

                        # Retrieve the argument role name from DWD Overlay.
                        for argument_role_id in argument_role_ids:
                            dwd_qnode = 'DWD_' + argument_role_id
                            try:
                                argument_role_name = dwd_overlay['relations'][dwd_qnode]['name']
                                argument_role_definition = dwd_overlay['relations'][dwd_qnode]['wd_description']
                            except:
                                try:
                                    argument_role_name = dwd_overlay['entities'][dwd_qnode]['name']
                                    argument_role_definition = dwd_overlay['entities'][dwd_qnode]['wd_description']
                                except:
                                    try:
                                        argument_role_name = dwd_overlay['events'][dwd_qnode]['name']
                                        argument_role_definition = dwd_overlay['events'][dwd_qnode]['wd_description']
                                    except:
                                        argument_role_name = "unknown"
                                        argument_role_definition = ""

                            argument_role = {'argument_type': 'VM', 'argument_role_name': argument_role_name, 'argument_role_definition': argument_role_definition}
                            event['argument_roles'].append(argument_role)

                    else:
                        raise ValueError("Invalid argmention_id!")
                    
                    events.append(event)

    # Build output.
    output = ""
    for i, event in enumerate(events):
        output += f"{i+1}. {event['event_mention']}. "
        if len(event['event_types']) <= 1:
            output += f"It's type is {event['event_types'][0]['event_type_name']}, defined as {event['event_types'][0]['event_type_definition']}. "
        else:
            output += f"It's types are "
            for j, et in enumerate(event['event_types']):
                if j < len(event['event_types'])-1:
                    output += f"{et['event_type_name']}, defined as {et['event_type_definition']}, and "
                else:
                    output += f"{et['event_type_name']}, defined as {et['event_type_definition']}. "
                    
        if len(event['arguments']) <= 1:
            output += "The argument of this event is "
        else:
            output += "The arguments of this event are "
            
        for k, arg in enumerate(event['arguments']):
            if k < len(event['arguments'])-1:
                if arg['argument_type'] == 'EM':
                    output += f"entity {arg['argument_mention']} playing the role {event['argument_roles'][k]['argument_role_name']}, defined as {event['argument_roles'][k]['argument_role_definition']}, and " 
                elif arg['argument_type'] == 'VM': # 'VM'
                    output += f"another event {arg['argument_mention']}. It's type is {event['argument_roles'][k]['argument_role_name']}, defined as {event['argument_roles'][k]['argument_role_definition']}, and "
                else: # 'RM'
                    output += f"a relation {arg['argument_mention']}. It's type is {event['argument_roles'][k]['argument_role_name']}, defined as {event['argument_roles'][k]['argument_role_definition']}, and "
            else: # the last argument
                if arg['argument_type'] == 'EM':
                    output += f"entity {arg['argument_mention']} playing the role {event['argument_roles'][k]['argument_role_name']}, defined as {event['argument_roles'][k]['argument_role_definition']}. "
                elif arg['argument_type'] == 'VM': # 'VM'
                    output += f"another event {arg['argument_mention']}. It's type is {event['argument_roles'][k]['argument_role_name']}, defined as {event['argument_roles'][k]['argument_role_definition']}. "
                else: # 'RM'
                    output += f"a relation {arg['argument_mention']}. It's type is {event['argument_roles'][k]['argument_role_name']}, defined as {event['argument_roles'][k]['argument_role_definition']}. "
        output += "\n"

    return output
                    

def get_relations(parent_uid, 
                  child_uid, 
                  claim_frames_df, 
                  evt_kes_df, 
                  evt_slots_df, 
                  rel_kes_df, 
                  rel_slots_df, 
                  arg_kes_df, 
                  dwd_overlay
                  ):
    """
    Retrieve relations for a document from the annotation data.
    Given a "parent_uid" of a document in parent_children.tab, find the "root_uid" (== "parent_uid") and 
    "associated_kes" in claim_frames.tab. Match them with the "root_uid" and "relationmention_id" in 
    rel_kes.tab and retrieve the "description" of the relation. Then, determine the relation type name and 
    its definition using the "qnode_type_id" and the DWD Overlay. After that, retrieve the "argmention_id" 
    in rel_slots.tab using "root_uid" and "relationmention_id" as queries. Use the "root_uid" and "argmention_id"
    as queries to find the relation arguments by retrieving the "description" in arg_kes.tab. Also in arg_kes.tab, 
    get the "qnode_type_id" and use it to get the argument role in the DWD Overlay.
    """

    relations = []
    root_uid = parent_uid
    associated_kes = claim_frames_df[claim_frames_df['root_uid']==root_uid]['associated_kes'].values

    for kes in associated_kes:
        for ke in kes.split(','):
            # ke can start with 'VM' (event mention id) or 'RM' (relation mention id).
            # Here, for relation extraction, we only need those start with 'RM'.
            if ke.startswith('RM'):
                relation = dict()

                # Retrieve the relation description.
                relation_mention = rel_kes_df[(rel_kes_df['root_uid']==root_uid) & 
                                              (rel_kes_df['relationmention_id']==ke)]['description'].values[0]
                relation['relation_mention'] = relation_mention

                # Retrieve the qnode_type_ids.
                relation_type_ids = rel_kes_df[(rel_kes_df['root_uid']==root_uid) & 
                                               (rel_kes_df['relationmention_id']==ke)]['qnode_type_id'].values[0]
                relation_type_ids = relation_type_ids.strip().split(' | ')

                relation['relation_types'] = []
                # Retrieve the relation type name and definition from DWD Overlay.
                for relation_type_id in relation_type_ids:
                    dwd_qnode = 'DWD_' + relation_type_id
                    try:
                        relation_type_name = dwd_overlay['relations'][dwd_qnode]['name']
                        relation_type_definition = dwd_overlay['relations'][dwd_qnode]['wd_description']
                    except:
                        relation_type_name = "unknown"
                        relation_type_definition = ""

                relation_type = {'relation_type_name': relation_type_name, 'relation_type_definition': relation_type_definition}
                relation['relation_types'].append(relation_type)

                # Use "root_uid" and "relationmention_id" in rel_kes.tab to retrieve the "argmention_id" in rel_slots.tab.
                argmention_ids = rel_slots_df[(rel_slots_df['root_uid']==root_uid) & 
                                              (rel_slots_df['relationmention_id']==ke)]['argmention_id'].values
                
                relation['arguments'] = []
                relation['argument_roles'] = []

                # Retrieve the relation argument descriptions and argument roles from DWD Overlay.
                for argmention_id in argmention_ids:
                    # If the current relation argument is an entity ('EM'),
                    if argmention_id.startswith('EM'):
                        # Retrieve the relation argument description.
                        argument_mention = arg_kes_df[(arg_kes_df['root_uid']==root_uid) & 
                                                      (arg_kes_df['argmention_id']==argmention_id)]['description'].values[0]
                        # Retrieve the qnode_type_id.
                        argument_role_ids = arg_kes_df[(arg_kes_df['root_uid']==root_uid) & 
                                                       (arg_kes_df['argmention_id']==argmention_id)]['qnode_type_id'].values[0]
                        argument_role_ids = argument_role_ids.strip().split(' | ')

                        relation_argument = {'argument_type': 'EM', 'argument_mention': argument_mention}
                        relation['arguments'].append(relation_argument)

                        # Retrieve the argument role name and definition from DWD Overlay.
                        for argument_role_id in argument_role_ids:
                            dwd_qnode = 'DWD_' + argument_role_id
                            try:
                                argument_role_name = dwd_overlay['entities'][dwd_qnode]['name']
                                argument_role_definition = dwd_overlay['entities'][dwd_qnode]['wd_description']
                            except:
                                try:
                                    argument_role_name = dwd_overlay['events'][dwd_qnode]['name']
                                    argument_role_definition = dwd_overlay['events'][dwd_qnode]['wd_description']
                                except:
                                    try:
                                        argument_role_name = dwd_overlay['relations'][dwd_qnode]['name']
                                        argument_role_definition = dwd_overlay['relations'][dwd_qnode]['wd_description']
                                    except:
                                        argument_role_name = "unknown"
                                        argument_role_definition = ""
                                        
                            argument_role = {
                                'argument_type': 'EM', 
                                'argument_role_name': argument_role_name, 
                                'argument_role_definition': argument_role_definition
                                }
                            relation['argument_roles'].append(argument_role)

                    # If the current relation argument is an event ('VM'),
                    elif argmention_id.startswith('VM'):
                        # Retrieve the event description (event mention) as the argument mention.
                        argument_mention = evt_kes_df[(evt_kes_df['root_uid']==root_uid) & 
                                                      (evt_kes_df['eventmention_id']==argmention_id)]['description'].values[0]
                        relation_argument = {'argument_type': 'VM', 'argument_mention': argument_mention}
                        relation['arguments'].append(relation_argument)

                        # Retrieve the relation type as argument role.
                        # Retrieve the qnode_type_ids.
                        argument_role_ids = evt_kes_df[(evt_kes_df['root_uid']==root_uid) & (evt_kes_df['eventmention_id']==argmention_id)]['qnode_type_id'].values[0]
                        argument_role_ids = argument_role_ids.strip().split(' | ')

                        for argument_role_id in argument_role_ids:
                            dwd_qnode = 'DWD_' + argument_role_id
                            try:
                                argument_role_name = dwd_overlay['relations'][dwd_qnode]['name']
                                argument_role_definition = dwd_overlay['relations'][dwd_qnode]['wd_description']
                            except:
                                try:
                                    argument_role_name = dwd_overlay['entities'][dwd_qnode]['name']
                                    argument_role_definition = dwd_overlay['entities'][dwd_qnode]['wd_description']
                                except:
                                    try:
                                        argument_role_name = dwd_overlay['events'][dwd_qnode]['name']
                                        argument_role_definition = dwd_overlay['events'][dwd_qnode]['wd_description']
                                    except:
                                        argument_role_name = "unknown"
                                        argument_role_definition = ""
                                        
                            argument_role = {'argument_type': 'VM', 
                                             'argument_role_name': argument_role_name, 
                                             'argument_role_definition': argument_role_definition
                                             }
                            relation['argument_roles'].append(argument_role)

                    # If the current event argument is another relation ('RM'),
                    elif argmention_id.startswith('RM'):
                        # Retrieve the relation description as the argument mention.
                        argument_mention = rel_kes_df[(rel_kes_df['root_uid']==root_uid) & (rel_kes_df['relationmention_id']==argmention_id)]['description'].values[0]
                        relation_argument = {'argument_type': 'RM', 'argument_mention': argument_mention}
                        relation['arguments'].append(relation_argument)
                        # Retrieve the relation type as the argument role.
                        # Retrieve the qnode_type_ids.
                        argument_role_ids = rel_kes_df[(rel_kes_df['root_uid']==root_uid) & (rel_kes_df['relationmention_id']==argmention_id)]['qnode_type_id'].values[0]
                        argument_role_ids = argument_role_ids.strip().split(' | ')

                        # Retrieve the argument role name and definition from DWD Overlay.
                        for argument_role_id in argument_role_ids:
                            dwd_qnode = 'DWD_' + argument_role_id
                            try:
                                argument_role_name = dwd_overlay['relations'][dwd_qnode]['name']
                                argument_role_definition = dwd_overlay['relations'][dwd_qnode]['wd_description']
                            except:
                                try:
                                    argument_role_name = dwd_overlay['entities'][dwd_qnode]['name']
                                    argument_role_definition = dwd_overlay['entities'][dwd_qnode]['wd_description']
                                except:
                                    try:
                                        argument_role_name = dwd_overlay['events'][dwd_qnode]['name']
                                        argument_role_definition = dwd_overlay['events'][dwd_qnode]['wd_description']
                                    except:
                                        argument_role_name = "unknown"
                                        argument_role_definition = ""

                            argument_role = {
                                'argument_type': 'RM', 
                                'argument_role_name': argument_role_name, 
                                'argument_role_definition': argument_role_definition
                                }
                            relation['argument_roles'].append(argument_role)

                    else:
                        raise ValueError("Invalid argument_id!")
                    
                relations.append(relation)

    # Build output.
    output = ""
    for i, relation in enumerate(relations):
        output += f"{i+1}. {relation['relation_mention']}. "
        if len(relation['relation_types']) <= 1:
            output += f"It's type is {relation['relation_types'][0]['relation_type_name']}, defined as {relation['relation_types'][0]['relation_type_definition']}"
        else:
            output += f"It's types are "
            for j, rt in enumerate(relation['relation_types']):
                if j < len(relation['relation_types'])-1:
                    output += f"{rt['relation_type_name']}, defined as {rt['relation_type_definition']}, and "
                else:
                    output += f"{rt['relation_type_name']}, defined as {rt['relation_type_definition']}. "
                    
        if len(relation['arguments']) <= 1:
            output += "The argument of this relation is "
        else:
            output += "The arguments of this relation are "
    
        for k, arg in enumerate(relation['arguments']):
            if k < len(relation['arguments'])-1:
                if arg['argument_type'] == 'EM':
                    output += f"entity {arg['argument_mention']} playing the role {relation['argument_roles'][k]['argument_role_name']}, defined as {relation['argument_roles'][k]['argument_role_definition']}, and "
                elif arg['argument_type'] == 'VM':
                    output += f"an event {arg['argument_mention']}. It's type is {relation['argument_roles'][k]['argument_role_name']}, defined as {relation['argument_roles'][k]['argument_role_definition']}, and "
                else: # 'RM'
                    output += f"another relation {arg['argument_mention']}. It's type is {relation['argument_roles'][k]['argument_role_name']}, defined as {relation['argument_roles'][k]['argument_role_definition']}, and "
            else: # the last argument
                if arg['argument_type'] == 'EM':
                    output += f"entity {arg['argument_mention']} playing the role {relation['argument_roles'][k]['argument_role_name']}, defined as {relation['argument_roles'][k]['argument_role_definition']}. "
                elif arg['argument_type'] == 'VM':
                    output += f"an event {arg['argument_mention']}. It's type is {relation['argument_roles'][k]['argument_role_name']}, defined as {relation['argument_roles'][k]['argument_role_definition']}. "
                else: # 'RM'
                    output += f"another relation {arg['argument_mention']}. It's type is {relation['argument_roles'][k]['argument_role_name']}, defined as {relation['argument_roles'][k]['argument_role_definition']}. "
        output += "\n"
        
    return output


def get_claims(parent_uid, 
               child_uid, 
               claim_frames_df, 
               evt_kes_df, 
               evt_slots_df, 
               rel_kes_df, 
               rel_slots_df, 
               arg_kes_df, 
               dwd_overlay
               ):
    """
    Retrieve claims for a document from the annotation data.
    A claim consists of following 12 components:
        1. DocID (parent_uid == root_uid)
        2. ClaimID
        3. Topic (and Subtopic)
        4. Claim Template
        5. X Variable (aka. claim object)
        6. Claimer
        7. Epistemic Status (i.e. compositional stance toward truth AND certainty)
        8. Affiliation
        9. Sentiment Status
        10. Date/Time
        11. Location
        12. Medium
        
    1 and 2 are fixed and do not need to be generated. The model should predict (i.e. generate) 3-12.
    In addition, the model also needs to generate the claim sentence, i.e. the description in claim_frames.tab.
    """

    claims = []
    # 1. DocID
    root_uid = parent_uid

    for _, row in claim_frames_df[claim_frames_df['root_uid']==parent_uid].iterrows():
        # 2. ClaimID
        claim_id = row['claim_id']
        
        # Claim Sentence
        claim_sentence = row['description']
        
        # 3. Topic and Subtopic
        topic = row['topic']
        subtopic = row['subtopic']
    
        # 4. Claim Template
        claim_template = row['claim_template']
    
        # 5. X Variable
        x_variable = row['x_variable']
    
        # Populated claim template (i.e. core claim)
        populated_claim_template = claim_template.replace("[X]", x_variable)
    
        # 6. Claimer
        claimer = row['claimer']
    
        # 7. Epistemic Status
        epistemic_status = row['epistemic_status']
        if epistemic_status != "unknown":
            epistemic_status_truth = epistemic_status.split('-')[0]
            epistemic_status_certainty = epistemic_status.split('-')[1]
        else:
            epistemic_status_truth = "unknown"
            epistemic_status_certainty = "unknown"
    
        # 8. Affiliation
        claimer_affiliation = row['claimer_affiliation']
    
        # 9. Sentiment Status
        sentiment_status = row['sentiment_status']
        
        # 10. Date/Time
        claim_datetime = row['claim_datetime']
        
        # 11. Location
        claim_location = row['claim_location']
        
        # 12. Medium
        claim_medium = row['claim_medium']
    
    
        claim = {
            "root_uid": root_uid, 
            "claim_id": claim_id, 
            "claim_sentence": claim_sentence, 
            "topic": topic, 
            "subtopic": subtopic, 
            "claim_template": claim_template, 
            "x_variable": x_variable, 
            "populated_claim_template": populated_claim_template, 
            "claimer": claimer, 
            "epistemic_status": epistemic_status, 
            "epistemic_status_truth": epistemic_status_truth, 
            "epistemic_status_certainty": epistemic_status_certainty, 
            "claimer_affiliation": claimer_affiliation, 
            "sentiment_status": sentiment_status, 
            "claim_datetime": claim_datetime, 
            "claim_location": claim_location, 
            "claim_medium": claim_medium
        }
        
        claims.append(claim)
    
    
    # Build output.
    output = ""
    for i, claim in enumerate(claims):
        output += f"{i+1}. {claim['claim_sentence']}. "
        output += f"It's about the topic {claim['topic']}. It's about the subtopic {claim['subtopic']}. "
        output += f"The object being claimed in the claim sentence is {claim['x_variable']}. "
        output += f"The claimer of this claim is {claim['claimer']}. "
        
        if claim['epistemic_status'] == "true-certain":
            output += f"The claimer asserts this claim and thinks it is certainly true. Thus, the epistemic status of the claimer to this claim is true-certain. "
        elif claim['epistemic_status'] == "true-uncertain":
            output += f"The claimer thinks that this claim may be true, but is uncertain about it. Thus, the epistemic status of the claimer to this claim is true-uncertain. "
        elif claim['epistemic_status'] == "false-certain":
            output += f"The claimer refutes this claim and thinks it is certainly false. Thus, the epistemic status of the claimer to this claim is false-certain. "
        elif claim['epistemic_status'] == "false-uncertain":
            output += f"The claimer thinks that this claim may be false, but is uncertain about it. Thus, the epistemic status of the claimer to this claim is false-uncertain. "
        else:
            output += f"The claimer's epistemic status to this claim cannot be determined from the document, or the claimer neither asserts nor refutes this claim. "
            
        if claim['sentiment_status'] == "positive":
            output += f"The claimer demonstrates a positive attitude (e.g. opinion, evaluation, emotion etc.) towards the claim. Thus, the claimer's sentiment status to this claim is {claim['sentiment_status']}. "
        elif claim['sentiment_status'] == "negative":
            output += f"The claimer demonstrates a negative attitude (e.g. opinion, evaluation, emotion etc.) towards the claim. Thus, the claimer's sentiment status to this claim is {claim['sentiment_status']}. "
        elif claim['sentiment_status'] == "mixed":
            output += f"The claimer expresses both positive and negative sentiment towards this claim, or the claimer explicitly expresses mixed feelings towards this claim. This means that the claimer demonstrates a mixed attitude (e.g. opinion, evaluation, emotion etc.) towards the claim. Thus, the claimer's sentiment status to this claim is {claim['sentiment_status']}. "
        else:
            output += f"The claimer's attitude (e.g. opinion, evaluation, emotion etc.) towards the claim cannot be determined from the document, or this claim is an objective statement of fact without any evidence in the document for the sentiment of the claimer toward the claim. Thus, the claimer's sentiment status to this claim is {claim['sentiment_status']}"
            
        output += f"The claimer of this claim is affiliated with {claim['claimer_affiliation']}. "
        
        if claim['claim_datetime'] == "unknown":
            output += "The date/time at which this claim was made is unknown. "
        else:
            output += f"The claim was made at {claim['claim_datetime']}. "
        
        if claim['claim_location'] == "EMPTY_NA":
            output += f"The location where this claim was made cannot be determined from the document. Thus, the claim location is {claim['claim_location']}. "
        else:
            output += f"The claim was made at the location {claim['claim_location']}. "
    
        if claim['claim_medium'] == "EMPTY_NA":
            output += f"The medium (broadcast source) of this claim cannote be determined from the document. Thus, the claim medium is {claim['claim_medium']}. "
        else:
            output += f"The claim medium on which the claim was broadcast is {claim['claim_medium']}. "

        output += "\n"
    
    return output


def main():
    args = parse_arguments()
    dataset_path = Path(args.data_path)
    translated_rsd_path = Path(args.trans_rsd_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Path to raw datasets:", dataset_path)
    print("Path to the translated rsd files:", translated_rsd_path)
    print("Output directory:", output_dir)
    
    # Load childuid2translatedrsd.p
    childuid2translatedrsd = pickle.load(open(f'{translated_rsd_path}/childuid2translatedrsd.p', 'rb'))
    
    # Load parent_child.tab
    SOURCE_DATA_PATH = f"{dataset_path}/LDC/LDC2021E11_AIDA_Phase_3_Practice_Topic_Source_Data_V2.0"
    parent_children_df = pd.read_csv(f"{SOURCE_DATA_PATH}/docs/parent_children.tab", sep='\t')
    
    # Load claim_frames.tab
    ANNOTATION_PATH = f"{dataset_path}/LDC/LDC2021E16_AIDA_Phase_3_TA3_Practice_Topic_Annotation_V5.1"
    claim_frames_df = pd.read_csv(f"{ANNOTATION_PATH}/data/ta3_ldc/claim_frames.tab", sep='\t')
    
    # Load topic_list_prompt_questions.txt, which derives from the original topic_list.txt file with additional, 
    # designed prompt questions for the claim templates.
    topic_list_df = pd.read_csv(f"{SOURCE_DATA_PATH}/docs/topic_list_prompt_questions.txt", sep='\t')
    
    # Concatenate the 11 candidate topics to a string.
    topic_list = f""
    for topic in set(topic_list_df['topic']):
        topic_list += f"{topic}\n"
    
    # Concatenate the 31 candidate subtopics to a string.
    subtopic_list = f""
    for subtopic in set(topic_list_df['subtopic']):
        subtopic_list += f"{subtopic}\n"
        
    # Load event knowledge elements (evt_kes.tab).
    evt_kes_df = pd.read_csv(f"{ANNOTATION_PATH}/data/ta3_ldc/evt_kes.tab", sep='\t')
    
    # Load event slots (evt_slots.tab).
    evt_slots_df = pd.read_csv(f"{ANNOTATION_PATH}/data/ta3_ldc/evt_slots.tab", sep='\t')
    
    # Load argument knowledge elements (arg_kes.tab).
    arg_kes_df = pd.read_csv(f"{ANNOTATION_PATH}/data/ta3_ldc/arg_kes.tab", sep='\t')
    
    # Load relation knowledge elements (rel_kes.tab).
    rel_kes_df = pd.read_csv(f"{ANNOTATION_PATH}/data/ta3_ldc/rel_kes.tab", sep='\t')
    
    # Load relation slots (rel_slots.tab).
    rel_slots_df = pd.read_csv(f"{ANNOTATION_PATH}/data/ta3_ldc/rel_slots.tab", sep='\t')
    
    # Load DWD Overlay.
    DWD_PATH = f"{dataset_path}/DWD_Overlay"
    dwd_overlay = json.load(open(f"{DWD_PATH}/xpo_v5.1a.json", 'r'))
    
    # Construct the dataset.
    # Dataset for Claim Frame Extraction (CFE)
    cfe = []

    # parent_uids that are in childuid2translatedrsd (i.e. in the source data) but not in the ta3_ldc annotation.
    parent_uids_not_in_ta3_ldc = []
    
    for child_uid_rsd, translated_rsd in tqdm(childuid2translatedrsd.items()):
        # Get the parent_uid in parent_children.tab using the child_uid in childuid2translatedrsd.
        parent_uid = parent_children_df[(parent_children_df['child_uid']==child_uid_rsd) & 
                                        (parent_children_df['child_asset_type']=='.ltf.xml')]['parent_uid'].values[0]
        
        if parent_uid not in list(claim_frames_df['root_uid']):
            parent_uids_not_in_ta3_ldc.append(parent_uid)
        else:
            # Create a mapping {'text': {'child_uid': <child_uid>, 'rsd_translated': <translated_rsd}}.
            text = {
                    'child_uid': child_uid_rsd,
                    'rsd_translated': translated_rsd
            }

            # Create a multi-turn conversation between human and robot.
            conversations = []

            preprompt_1 = {
                "speaker": "User",
                "content": """<s>[INST] <<SYS>> You are a helpful, respectful and honest assistant (<Assistant>) that is designed to complete the following task.\n Task description: Given a document, a list of topics and a list of subtopics, your final task is to extract all the claims from the article by answering questions of the user (<User>) or following the user's instructions. To achieve this, you'll have to first extract events and relations that comprise claims from the document, then infer the claims from them.\n\nConcept definitions:\n1. Event: This is a specific occurrence of an action or state change that happens in a certain time and a certain place, involving one or more participants (arguments) and the roles they play in the event. An event consists of a trigger that cause the event, the event type to which the event corresponds, arguments such as entities, non-entity participants, time and place, as well as the role played by an argument in the event.\n2. Relation: This is the relationship between two entities. A relation consists of a relation name, some entities and the roles they play in the relation.\n3. Claim: Each claim consists of a topic, a subtopic, a claim template populated by a variable called 'claim object', the claimer, the epistemic status of the claimer, the affiliation of the claimer, the sentiment of the claimer, the data/time of the claim, the location of the claim, and the medium of the claim.\n4. Topic and subtopic: The topic and subtopic are concise summarizations describing what the claim is about. \n5. Claim template: Each combination of a topic and a subtopic determines a claim template that includes the object of the claim. \n6. Claimer: This is the entity (person, organization etc.) that makes the claim. It is a name or a short English descriptive phrase that identifies the claimer as specifically as possible using information from the document. If the claimer is made by the unnamed document author, or if it is impossible to infer the claimer, then the claimer of that claim is set to the value 'author'.\n7. Epistemic status: This is a two-fold attribute of a claim. It represents the claimer's stance toward the truth and certainty of the claim. An epistemic status consists of the stance toward the truth, which is either 'true' or 'false', and the stance toward the certainty, which is either 'certain' or 'uncertain'. The output should be a combination of these two attributes. If the claimer's epistemic status cannot be determined from the document, then it is set to the value 'unknown'. Thus, the answer to the epistemic status should be one of the following values: true-certain, true-uncertain, false-certain, false-uncertain, unknown.\n8. Sentiment status: This is a single-value attribute of a claim. It represents the claimer's sentiment toward the claim. The answer to the sentiment status should be one of the following values: positive, negative, mixed, neutral-unknown.\n9. Date/Time: This is the date and time at which the claim was made. The answer to the date/time should be conformed with the format [before | after | on ] [YYYY-MM-DDTHH:MM:SS], where YYYY is the year, the first MM is the month, DD is the day, T is a fixed separator, HH is the hour, the second MM is the minute, SS is the second. If the date/time of the claim is unknown in the document, the output is set to the value 'unknown'.\n10. Affiliation: This is the entity (organization, nationality etc.) that the claimer is affiliated with. It is a name or a short English descriptive phrase that identifies the claimer affiliation as specifically as possible using information from the document.\n11. Location: This is the place where the claim was made. It is a name or a short English descriptive phrase that identifies the location as specifically as possible using information from the document. If the location cannot be determined from the document, then it is set to the value 'EMTPY_NA'.\n12. Medium: This is the broadcast source on where the claim was made. It should be a name or a short English descriptive phrase that identifies the medium as specifically as possible using information from the document. If the medium cannot be determined from the document, then it is set to the value 'EMPTY_NA'.\n\nEmphasis & Caution: Try your best to make sure that your answers can be found in the article. It is allowed to use 'EMPTY_NA' if you cannot find them in the document or infer from the document. But try to avoid that as best as possible.<</SYS>>\n\n<User>Do you understand your task?\n\n[/INST]"""
            }
            conversations.append(preprompt_1)

            preprompt_2 = {
                "speaker": "Assistant",
                "content": "<Assistant> Understood! I'll do my best to assist you in extracting the claims with all their components from the document, keeping the format of my answers conform with the format you specified. </s>"
            }
            conversations.append(preprompt_2)

            # Add the document and a turn about event extraction to the conversation.
            prompt_ee_human = f"<s>[INST] <User> Given the following document:\n{translated_rsd}\n\nWhat events are happening or happened in this document? What are the types of these events? What are the arguments of these events and their roles? Answer these questions one by one.\n\n[/INST]"
            turn_ee_human = {
                "speaker": "User",
                "content": prompt_ee_human
            }
            conversations.append(turn_ee_human)

            # Extract events.
            events = get_events(parent_uid, child_uid_rsd, claim_frames_df, evt_kes_df, evt_slots_df, rel_kes_df, rel_slots_df, arg_kes_df, dwd_overlay)
        #     print('EVENTS:\n', events, '\n\n')
            prompt_ee_robot = f"<Assistant> Following events are happening or happend in this document:\n{events}\n\n</s>"
            turn_ee_robot = {
                "speaker": "Assistant",
                "content": prompt_ee_robot
            }
            conversations.append(turn_ee_robot)

            # Add a turn about relation extraction to the conversation.
            prompt_re_human = f"<s>[INST] <User> What are the important relations in this document? What are the types of these relations? Which entities are connected by these relations? What are the roles of these entities? Answer these questions one by one.\n\n[/INST]" 
            turn_re_human = {
                "speaker": "User",
                "content": prompt_re_human
            }
            conversations.append(turn_re_human)

            # Extract relations.
            relations = get_relations(parent_uid, child_uid_rsd, claim_frames_df, evt_kes_df, evt_slots_df, rel_kes_df, rel_slots_df, arg_kes_df, dwd_overlay)
            prompt_re_robot = f"<Assistant> This document contains following relations:\n{relations}\n\n</s>"
            turn_re_robot = {
                "speaker": "Assistant",
                "content": prompt_re_robot
            }
            conversations.append(turn_re_robot)

            # Add a turn about claim extraction to the conversation.
            prompt_ce_human = f"<s>[INST] <User> Given the following list of candidate topics {topic_list} and list of subtopics {subtopic_list}. Which claims do these events and relations comprise?\n\n[/INST]"
            turn_ce_human = {
                "speaker": "User",
                "content": prompt_ce_human
            }
            conversations.append(turn_ce_human)

            # Extract claims.
            claims = get_claims(parent_uid, 
                                child_uid_rsd, 
                                claim_frames_df, 
                                evt_kes_df, 
                                evt_slots_df, 
                                rel_kes_df, 
                                rel_slots_df, 
                                arg_kes_df, 
                                dwd_overlay
                                )
            prompt_ce_robot = f"<Assistant> These events and relations comprise the following claims with their correpsonding components:\n{claims}\n\n</s>"
            turn_ce_robot = {
                "speaker": "Assistant",
                "content": prompt_ce_robot
            }
            conversations.append(turn_ce_robot)

            cfe.append(
                {
                    'parent_uid': parent_uid, 
                    'text': text, 
                    'conversations': conversations,
                }
            )

            with open(f"{output_dir}/instruction_data.json", 'w') as file:
                json.dump(cfe, file)


if __name__ == "__main__":
    main()