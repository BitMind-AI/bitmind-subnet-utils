
import wandb
from collections import defaultdict
import pandas as pd
import os
import datetime
from typing import List, Dict, Any, Optional, Union, DefaultDict, Set

from metrics import compute_metrics


def get_wandb_runs(
        entity: str,
        project: str,
        start_ts: Union[int, float, datetime.datetime, str],
        end_ts: Optional[Union[int, float, datetime.datetime, str]] = None,
        validator_run_name: Optional[str] = None) -> List[Any]:
    """
    Get W&B runs for a specific project with filtering options.
    
    Args:
        entity: W&B entity (username or team name)
        project: W&B project name
        start_ts: Start timestamp/datetime for filtering runs
        end_ts: Optional end timestamp/datetime for filtering runs
        validator_run_name: Optional filter for a specific validator run name
    
    Returns:
        List of W&B run objects
    
    Notes:
        Must run `wandb login` and provide your W&B API key when prompted
    """
    # Standardize datetime inputs
    if isinstance(start_ts, datetime.datetime):
        start_ts = start_ts.isoformat()
    elif isinstance(start_ts, (int, float)):
        start_ts = datetime.datetime.fromtimestamp(start_ts).isoformat()
        
    if end_ts is not None:
        if isinstance(end_ts, datetime.datetime):
            end_ts = end_ts.isoformat()
        elif isinstance(end_ts, (int, float)):
            end_ts = datetime.datetime.fromtimestamp(end_ts).isoformat()
    
    # Build filters
    filters = {}
    if validator_run_name:
        filters["display_name"] = validator_run_name

    filters["created_at"] = {"$gte": start_ts}
    if end_ts:
        filters["created_at"]["$lte"] = end_ts

    print("Querying W&B with filters:", filters)
    api = wandb.Api()
    return api.runs(f"{entity}/{project}", filters=filters)


def get_unique_validator_run_names(
        entity: str,
        project: str,
        start_ts: Union[int, float, datetime.datetime, str],
        end_ts: Optional[Union[int, float, datetime.datetime, str]] = None) -> Set[str]:
    """
    Get set of unique validator run names for a project within a time range.
    
    Args:
        entity: W&B entity (username or team name)
        project: W&B project name
        start_ts: Start timestamp/datetime for filtering runs
        end_ts: Optional end timestamp/datetime for filtering runs
    
    Returns:
        Set of unique run names
    """
    # Standardize datetime inputs
    if isinstance(start_ts, datetime.datetime):
        start_ts = start_ts.isoformat()
    elif isinstance(start_ts, (int, float)):
        start_ts = datetime.datetime.fromtimestamp(start_ts).isoformat()
        
    if end_ts is not None:
        if isinstance(end_ts, datetime.datetime):
            end_ts = end_ts.isoformat()
        elif isinstance(end_ts, (int, float)):
            end_ts = datetime.datetime.fromtimestamp(end_ts).isoformat()
    
    # Build filters
    filters = {"created_at": {"$gte": start_ts}}
    if end_ts:
        filters["created_at"]["$lte"] = end_ts

    print(f"Querying {entity}/{project} with filters:", filters)
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters=filters)

    return {run.name for run in runs}


def compute_miner_performance(
        wandb_validator_runs: List[Any],
        miner_uids: Optional[List[str]] = None, 
        start_ts: Optional[Union[int, float, datetime.datetime]] = None,
        end_ts: Optional[Union[int, float, datetime.datetime]] = None,
        validator_run_name: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Compute performance metrics for miners from W&B validator runs.
    
    Args:
        wandb_validator_runs: List of W&B runs
        miner_uids: Optional list of miner UIDs to filter for
        start_ts: Optional start timestamp for filtering challenges
        end_ts: Optional end timestamp for filtering challenges
        validator_run_name: Optional filter for a specific validator run
    
    Returns:
        Dictionary containing prediction dataframe and performance metrics dataframe
    """
    # Standardize datetime inputs
    if isinstance(start_ts, datetime.datetime):
        start_ts = start_ts.timestamp()
    if isinstance(end_ts, datetime.datetime):
        end_ts = end_ts.timestamp()
    
    challenge_data: DefaultDict[str, List[Any]] = defaultdict(list)
    
    for run in wandb_validator_runs:
        if validator_run_name is not None and run.name != validator_run_name: 
            continue

        history_df = run.history()
        for _, challenge_row in history_df.iterrows():
            if start_ts is not None and challenge_row['_timestamp'] < start_ts:
                continue
            if end_ts is not None and challenge_row['_timestamp'] > end_ts:
                continue

            try:
                miner_preds = challenge_row['pred']
            except KeyError:
                miner_preds = challenge_row['predictions']
                
            try:
                challenge_miner_uids = challenge_row['miner_uid']
            except KeyError:
                challenge_miner_uids = challenge_row['miner_uids']
                
            if isinstance(challenge_miner_uids, dict):  
                continue  # ignore improperly formatted instances
            
            modality = challenge_row.get('modality', 'image')
            label = challenge_row['label']
            
            # record predictions and labels for each miner
            for pred, uid in zip(miner_preds, challenge_miner_uids):
                if miner_uids is not None and uid not in miner_uids:
                    continue
                if pred == -1:
                    continue
                challenge_data['modality'].append(modality)
                challenge_data['uid'].append(uid)
                challenge_data['prediction'].append(pred)
                challenge_data['label'].append(label)

                try:
                    challenge_data['wandb_filepath'].append(challenge_row.get(modality, 'video_1')['path'])
                except Exception:
                     challenge_data['wandb_filepath'].append('No Media Found')

                challenge_data['validator_run'].append(run.name)
                challenge_data['timestamp'].append(challenge_row['_timestamp'])

    all_miner_preds_df = pd.DataFrame(challenge_data)

    # Compute performance metrics for each miner
    miner_perf_data = []
    for uid, miner_preds in all_miner_preds_df.groupby('uid'):
        for modality in ['image', 'video']:
            miner_modality_preds = miner_preds[miner_preds['modality'] == modality]
            if len(miner_modality_preds) > 0:
                metrics = compute_metrics(
                    miner_modality_preds['prediction'].tolist(), 
                    miner_modality_preds['label'].tolist())
                metrics['uid'] = uid
                metrics['modality'] = modality
                miner_perf_data.append(metrics)
    
    miner_perf_df = pd.DataFrame(miner_perf_data)
    return {'predictions': all_miner_preds_df, 'performance': miner_perf_df}

def download_challenge_media(
        wandb_validator_runs: List[Any],
        download_dest: str = '',
        download_images: bool = True,
        download_videos: bool = True,
        download_limit: Optional[int] = None,
        miner_uids: Optional[List[str]] = None,
        start_ts: Optional[Union[int, float, datetime.datetime]] = None,
        end_ts: Optional[Union[int, float, datetime.datetime]] = None,
        validator_run_name: Optional[str] = None,
        verbose: bool = True) -> pd.DataFrame:
    """
    Download images and videos from W&B validator runs and return a dataframe with filepaths.
    
    Args:
        wandb_validator_runs: List of W&B runs
        download_dest: Destination directory for downloads
        download_images: Whether to download images
        download_videos: Whether to download videos
        download_limit: Maximum number of files to download
        miner_uids: Optional list of miner UIDs to filter by
        start_ts: Optional start timestamp for filtering challenges
        end_ts: Optional end timestamp for filtering challenges
        validator_run_name: Optional filter for a specific validator run
        verbose: Whether to print download progress messages
    
    Returns:
        Dataframe containing challenge and media information including local filepaths
    """
    # Standardize datetime inputs
    if isinstance(start_ts, datetime.datetime):
        start_ts = start_ts.timestamp()
    if isinstance(end_ts, datetime.datetime):
        end_ts = end_ts.timestamp()
        
    # Create destination directory if it doesn't exist
    if download_dest and not os.path.exists(download_dest):
        os.makedirs(download_dest)
        
    download_data: DefaultDict[str, List[Any]] = defaultdict(list)
    downloaded = 0
    
    for run in wandb_validator_runs:
        if validator_run_name is not None and run.name != validator_run_name: 
            continue

        history_df = run.history()
        for _, challenge_row in history_df.iterrows():
            if start_ts is not None and challenge_row['_timestamp'] < start_ts:
                continue
            if end_ts is not None and challenge_row['_timestamp'] > end_ts:
                continue

            try:
                challenge_miner_uids = challenge_row['miner_uid']
            except KeyError:
                challenge_miner_uids = challenge_row['miner_uids']
                
            if isinstance(challenge_miner_uids, dict):  
                continue  # ignore improperly formatted instances
                
            # Skip if we're filtering by miner UIDs and none of the UIDs match
            if miner_uids is not None and not any(uid in miner_uids for uid in challenge_miner_uids):
                continue
                
            modality = challenge_row.get('modality', 'image')
            
            # Process the media file
            should_download = ((modality == 'image' and download_images) or 
                               (modality == 'video' and download_videos))
            
            if should_download and (not download_limit or downloaded < download_limit):
                try:
                    media_path = challenge_row[modality]['path']
                    filename = os.path.basename(media_path)
                    local_path = os.path.join(download_dest, media_path)
                    if not os.path.exists(local_path):
                        if verbose:
                            print(f"Downloading {modality}: {media_path}")
                        run.file(media_path).download(download_dest)
                        downloaded += 1
                    else:
                        if verbose:
                            print(f"File already exists: {local_path}")
                    
                    # Record information for all miners involved
                    for uid in challenge_miner_uids:
                        if miner_uids is not None and uid not in miner_uids:
                            continue
                            
                        download_data['validator_run'].append(run.name)
                        download_data['modality'].append(modality)
                        download_data['uid'].append(uid)
                        download_data['wandb_filepath'].append(media_path)
                        download_data['local_filepath'].append(local_path)
                        download_data['timestamp'].append(challenge_row['_timestamp'])
                        
                except Exception as e:
                    if verbose:
                        print(f'Failed to download {modality}: {e}')
            else:
                # Even if we don't download, record the information
                for uid in challenge_miner_uids:
                    if miner_uids is not None and uid not in miner_uids:
                        continue
                        
                    try:
                        media_path = challenge_row.get(modality, 'video_1')['path']
                        local_path = os.path.join(download_dest, os.path.basename(media_path))
                        local_path = local_path if os.path.exists(local_path) else 'not downloaded'

                    except: 
                        media_path = ''
                        local_path = 'Download Failed'
                        download_data['validator_run'].append(run.name)
                        download_data['modality'].append(modality)
                        download_data['uid'].append(uid)
                        download_data['timestamp'].append(challenge_row['_timestamp'])
                        download_data['wandb_filepath'].append(media_path)
                        download_data['local_filepath'].append(local_path)
    
    return pd.DataFrame(download_data)


def merge_performance_and_downloads(
        predictions_df: pd.DataFrame,
        download_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Merge the performance results with the download dataframe.
    
    Args:
        perf_results: Dictionary containing 'predictions' and 'performance' dataframes
        download_df: Dataframe with download information
    
    Returns:
        Dictionary with merged prediction dataframe and the original performance dataframe
    """
    if download_df.empty:
        return predictions_df
        

    predictions_df['join_key'] = predictions_df['wandb_filepath'] + '|' + predictions_df['uid'].astype(str)
    download_df['join_key'] = download_df['wandb_filepath'] + '|' + download_df['uid'].astype(str)
    
    merged_df = pd.merge(
        predictions_df,
        download_df[['join_key', 'local_filepath']],
        on='join_key',
        how='left',
        suffixes=('', '_download')
    )
    merged_df['local_filepath'] = merged_df['local_filepath'].fillna('not downloaded')
    return merged_df.drop(columns=['join_key'])
    


