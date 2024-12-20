#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import itertools
from IPython.display import HTML


# In[2]:


logs = {}
logs['DubinsCar_preloop'] = ['/home/jeappen/code/PyCharm_deploy/gcbfplus/logs/DubinsCar/summary_5_30_preloop.csv',
                            '/home/jeappen/code/PyCharm_deploy/gcbfplus/pretrained/DubinsCar/gcbf+/test_log_5_30_pre_loop.csv']


# ## Read the STLPy test data into a table 

# In[3]:

def load_df(log_keys=None):
    if log_keys is None:
        log_keys = ['DubinsCar_preloop']
    all_dfs = [pd.read_csv(csv_file_2load) for log_key in log_keys for csv_file_2load in logs[log_key]]
    df = pd.concat(all_dfs, ignore_index=True)
    return df

# In[4]:

def dummy_ode_data(df):
    # Create dummy ode columns for now
    ode_dummy = df.loc[df['planner']=='gnn-ode'].copy()

    ode_dummy['planner'] = 'ode'
    ode_dummy['finish_mean'] = 0
    ode_dummy['safe_mean'] = 0
    return pd.concat([df, ode_dummy], ignore_index=True)


# In[5]:



# In[ ]:

# pd.to_numeric(df['TtR'],errors='coerce')


# In[7]:


spec_map = (('seq', 'Sequence'), ('cover', 'Cover'),('2loop', 'Loop'), ('2branch', 'Branch'))
n_goals = (3,3,2,2)
spec_replacement_dict = {f"{spec}{n}":f"{Spec}" for (spec,Spec),n in zip(spec_map,n_goals)}
# spec_replacement_dict = {f"{spec}{n}":f"{Spec}{n}" for (spec,Spec),n in itertools.product(spec_map,n_goals)}

def make_clickable(val):
    # target _blank to open new window
    return '<a target="_blank" href="{}">{}</a>'.format(val, 'Video')

def add_formatted_cols(df):
    def add_stddev(df, new_col, col1, col2):
        def combine_columns(row):
            return f"{row[col1]:0.2f}"#u"\u00B1"f"{row[col2]:2.2f}"  # Example 

        df[new_col] = df.apply(combine_columns, axis=1)
        return df

    # Cols without mean in key
    cols_id = ['finish_rate','mean_path_score', 'success_mean']
    cols = ['Finish Rate ↑', 'Mean Score ↑', 'Success Rate ↑']
    for new_col, col_name in zip(cols, cols_id):
        col1 = col_name
        col2 = f"{col_name}_std"
        add_stddev(df, new_col, col1, col2)

    # Cols with mean in key
    cols_id = ['safe','plan_time']
    cols = ['Safety Rate ↑', 'Planning Time ↓']
    for new_col, col_name in zip(cols, cols_id):
        col1 = f"{col_name}_mean"
        col2 = f"{col_name}_std"
        add_stddev(df, new_col, col1, col2)
        
    # Format any remaining cols
    
    df['Planning Time (s) ↓'] = df.apply(lambda row: f"{row['plan_time_mean']:0.2f}", axis=1)
    def format_if_not_nan(val):
        if pd.isna(float(val)):
            return "-"
        else:
            return f"{val:0.2f}"
    df['TtR ↓'] = df.apply(lambda row: format_if_not_nan(float(row['TtR'])), axis=1)
    df['Planner'] = df.apply(lambda row: f"{row['planner'].upper()}", axis=1)
    
    df['Spec'] = df['spec'].map(spec_replacement_dict)
    df['Video'] = "https://www.google.com/"
    df.style.format({'Video': make_clickable})
    df['Obs'] = df.apply(lambda row: 'Y' if row['n_obs']>0 else 'N', axis=1)
    df['N'] = df['num_agents']




# In[8]:




# In[9]:


def create_pivoted_table(df):
    # Can show path to find correct files
    df_pivoted = df.pivot_table(index=['Spec', 'N', 'Obs'], columns='Planner', values= ['Finish Rate ↑', 'Safety Rate ↑', 'Mean Score ↑', 'Planning Time (s) ↓','TtR ↓', 'Success Rate ↑'],#, 'Video'],#, 'path'],
                               aggfunc='first')
#     df_pivoted = df_grouped.first().unstack(level='planner')
    df_pivoted.columns = df_pivoted.columns.swaplevel(0, 1)
    df_pivoted = df_pivoted.sort_index(axis=1, level=0)
    return df_pivoted

def replace_latex_math(string_to_replace):
    """Simple fn to replace given symbols with the latex alternatives"""
    symbols_to_replace = ['↓', '↑', r'\cline']
    latex_code = ['$\downarrow$', r'$\uparrow$' , r'\cmidrule']
    for symbol, latex in zip(symbols_to_replace, latex_code):
        string_to_replace = string_to_replace.replace(symbol, latex)

    return string_to_replace

def create_final_table():
    df = load_df()
    # Only filter rows with async_planner true
    df = df.loc[df['async_planner']].copy()

    add_formatted_cols(df)


    final_table =  create_pivoted_table(df)

    all_planners = final_table.keys().get_level_values('Planner').unique().to_list()
    print(all_planners)

    # To select a few specs out of whole list

    specs = list(spec_replacement_dict.values())
    final_ind = [ind for ind in final_table.index if ind[0] in specs]

    formatted_table = final_table.loc[final_ind]

    final_table = formatted_table.copy()

    # Latex version
    final_latex = replace_latex_math(final_table.to_latex())
    print(final_latex)

    # HTML version
    # Trying to keep borders in final table
    # for k in all_planners:
    #     final_table[(k,'Video')]=final_table[(k,'Video')].apply(make_clickable)
    # final_html = formatted_table.style.format({(k,'Video'): make_clickable for k in all_planners}).to_html()
    # Prefer below since borders are lost in previous version
    final_html = final_table.to_html().replace("&lt;", "<").replace("&gt;", ">")


if __name__ == '__main__':
    create_final_table()