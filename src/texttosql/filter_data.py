"""
The initial column contains 486 columns. It reduces to 86 columns
"""

import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

column_list = ['source_file', 'source_network', 'source_proposal_type',
       'source_row_id', 'allowedcommentor', 'content', 'createdat',
       'datasource', 'hash', 'history_0_content',
       'history_0_createdat_nanoseconds', 'history_0_createdat_seconds',
       'history_0_title', 'history_1_content',
       'history_1_createdat_nanoseconds', 'history_1_createdat_seconds',
       'history_1_title', 'history_2_content',
       'history_2_createdat_nanoseconds', 'history_2_createdat_seconds',
       'history_2_title', 'id', 'index', 'isdefaultcontent', 'isdeleted',
       'linkedpost_indexorhash', 'linkedpost_proposaltype', 'metrics_comments',
       'metrics_reactions_dislike', 'metrics_reactions_like', 'network',
       'onchaininfo_beneficiaries_0_address',
       'onchaininfo_beneficiaries_0_amount',
       'onchaininfo_beneficiaries_0_assetid', 'onchaininfo_createdat',
       'onchaininfo_curator', 'onchaininfo_decisionperiodendsat',
       'onchaininfo_description', 'onchaininfo_hash', 'onchaininfo_index',
       'onchaininfo_origin', 'onchaininfo_prepareperiodendsat',
       'onchaininfo_proposer', 'onchaininfo_reward', 'onchaininfo_status',
       'onchaininfo_type', 'onchaininfo_votemetrics',
       'onchaininfo_votemetrics_aye_count',
       'onchaininfo_votemetrics_aye_value',
       'onchaininfo_votemetrics_bareayes_value',
       'onchaininfo_votemetrics_nay_count',
       'onchaininfo_votemetrics_nay_value',
       'onchaininfo_votemetrics_support_value', 'poll', 'proposaltype',
       'publicuser_addresses_0', 'publicuser_addresses_1',
       'publicuser_addresses_2', 'publicuser_addresses_3',
       'publicuser_addresses_4', 'publicuser_createdat', 'publicuser_id',
       'publicuser_profiledetails_bio', 'publicuser_profiledetails_coverimage',
       'publicuser_profiledetails_image',
       'publicuser_profiledetails_publicsociallinks_0_platform',
       'publicuser_profiledetails_publicsociallinks_0_url',
       'publicuser_profiledetails_title', 'publicuser_profilescore',
       'publicuser_rank', 'publicuser_username', 'tags_0_lastusedat',
       'tags_0_network', 'tags_0_value', 'tags_1_lastusedat', 'tags_1_network',
       'tags_1_value', 'tags_2_lastusedat', 'tags_2_network', 'tags_2_value',
       'title', 'topic', 'updatedat', 'userid', 'onchaininfo_beneficiaries_0_assetid', 'row_index']

governance_data_486 = pd.read_csv(str(os.getenv("BASE_PATH")) + "/onchain_data/onchain_first_pull/one_table/combined_governance_data.csv")
governance_data_86 = governance_data_486[column_list]

"""
Some of the columns are not present in the governance_data_86 dataframe
"""

print(f"Total number of records in the final CSV: {len(governance_data_86)}")
print(f"Total number of columns in the final CSV: {len(governance_data_86.columns)}")

governance_data_86.to_csv(str(os.getenv("BASE_PATH")) + "/onchain_data/onchain_first_pull/one_table/filter_data/governance_data_86.csv", index=False)