import os
import pandas as pd
import numpy as np
from typing import Optional, List, Union, Dict, Any
from datetime import datetime


def generate_media_gallery(
    df: pd.DataFrame,
    output_path: str = "media_gallery.html",
    title: str = "Media Gallery",
    max_items: Optional[int] = None,
    width: int = 500,
    height: int = 400,
    include_cols: Optional[List[str]] = None,
    exclude_cols: Optional[List[str]] = None,
    sort_by: Optional[Union[str, List[str]]] = None,
    ascending: Union[bool, List[bool]] = True
) -> str:
    """
    Generate an HTML gallery of images and videos with associated metadata,
    grouping multiple predictions for the same media.
    
    Args:
        df: DataFrame containing at least 'local_filepath', 'uid', and 'prediction' columns
        output_path: Path to save the HTML file
        title: Title for the HTML gallery
        max_items: Maximum number of unique media items to include (None for all)
        width: Maximum width for media display in pixels
        height: Maximum height for media display in pixels
        include_cols: List of columns to include in metadata (None for all)
        exclude_cols: List of columns to exclude from metadata (None for none)
        sort_by: Column(s) to sort by (None for no sorting)
        ascending: Whether to sort in ascending order
        
    Returns:
        Path to the generated HTML file
    """
    # Filter for only rows with valid local filepaths
    valid_df = df[df['local_filepath'] != 'not downloaded'].copy()
    
    if valid_df.empty:
        print("No valid media files found in the dataframe.")
        return None
    
    # Group by media file path to combine multiple predictions for the same media
    media_groups = []
    
    # Find unique media files while keeping track of all predictions
    for filepath, group_df in valid_df.groupby('local_filepath'):        # Create a record for this media file
        media_item = {
            'local_filepath': group_df['local_filepath'].iloc[0],
            'predictions': []
        }
        # Store all predictions for this media
        for _, row in group_df.iterrows():
            prediction_item = {}
            for col in group_df.columns:
                if col != 'local_filepath':
                    prediction_item[col] = row[col]
            media_item['predictions'].append(prediction_item)
        
        # Add some common metadata from the first row
        for col in group_df.columns:
            if col not in ['uid', 'prediction', 'local_filepath']:
                # For shared metadata, take the first value
                media_item[col] = group_df[col].iloc[0]
        
        media_groups.append(media_item)
    
    # Sort the grouped data if specified
    if sort_by is not None:
        if isinstance(sort_by, list):
            # Sort by the first available column in the list
            for col in sort_by:
                if col in media_groups[0]:
                    media_groups = sorted(media_groups, key=lambda x: x.get(col, 0), reverse=not ascending)
                    break
        else:
            # Sort by the specified column if it exists
            if sort_by in media_groups[0]:
                media_groups = sorted(media_groups, key=lambda x: x.get(sort_by, 0), reverse=not ascending)
    
    # Limit number of items if specified
    if max_items is not None:
        media_groups = media_groups[:max_items]
    
    print(f"Grouped {len(valid_df)} rows into {len(media_groups)} unique media items")
    
    # Determine which columns to show in the common metadata section
    all_cols = list(set().union(*[set(item.keys()) for item in media_groups]))
    all_cols = [col for col in all_cols if col not in ['local_filepath', 'predictions']]
    
    if include_cols is not None:
        metadata_cols = [col for col in include_cols if col in all_cols]
    elif exclude_cols is not None:
        metadata_cols = [col for col in all_cols if col not in exclude_cols 
                         and col not in ['uid', 'prediction']]
    else:
        # By default, include all columns except specific ones
        metadata_cols = [col for col in all_cols if col not in 
                        ['local_filepath', 'predictions', 'uid', 'prediction']]
    
    # Generate HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            h1 {{
                color: #333;
                text-align: center;
            }}
            .gallery {{
                display: flex;
                flex-direction: column;
                gap: 30px;
            }}
            .media-item {{
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                padding: 20px;
                display: flex;
                flex-direction: row;
                gap: 20px;
            }}
            .media-container {{
                min-width: {width}px;
            }}
            .media-container img, .media-container video {{
                max-width: {width}px;
                max-height: {height}px;
                border-radius: 4px;
                border: 1px solid #ddd;
            }}
            .metadata {{
                flex-grow: 1;
            }}
            h3 {{
                margin-top: 0;
                color: #444;
                border-bottom: 1px solid #eee;
                padding-bottom: 8px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}
            th, td {{
                padding: 8px 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            .predictions-table {{
                margin-top: 15px;
            }}
            .predictions-table th {{
                background-color: #e7f3ff;
            }}
            .correct-both {{
                background-color: #d4edda;  /* Green */
            }}
            .correct-one {{
                background-color: #fff3cd;  /* Yellow */
            }}
            .correct-none {{
                background-color: #f8d7da;  /* Red */
            }}
            .timestamp {{
                color: #666;
                font-size: 0.9em;
            }}
            .item-count {{
                color: #666;
                font-size: 0.8em;
                text-align: center;
                margin-bottom: 20px;
            }}
            @media (max-width: 900px) {{
                .media-item {{
                    flex-direction: column;
                }}
                .media-container {{
                    min-width: unset;
                    width: 100%;
                    display: flex;
                    justify-content: center;
                }}
            }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <p class="item-count">Showing {len(media_groups)} unique media items with {len(valid_df)} total predictions | Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <div class="gallery">
    """
    
    # Add each media item with its metadata
    for i, media_item in enumerate(media_groups):
        filepath = media_item['local_filepath']
        file_ext = os.path.splitext(filepath)[1].lower()
        html += f'<div class="media-item">\n'
        html += f'    <div class="media-container">\n'
        
        # Handle different media types
        if file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']:
            # Image files
            html += f'        <img src="{filepath}" alt="Image {i+1}">\n'
        elif file_ext in ['.mp4', '.webm', '.ogg', '.mov']:
            # Video files
            html += f'        <video controls>\n'
            html += f'            <source src="{filepath}" type="video/{file_ext[1:]}">\n'
            html += f'            Your browser does not support the video tag.\n'
            html += f'        </video>\n'
        elif file_ext == '.gif':
            # GIF files (treat as images)
            html += f'        <img src="{filepath}" alt="GIF {i+1}">\n'
        else:
            # Unknown file type
            html += f'        <p>Unsupported media type: {file_ext}</p>\n'
            html += f'        <p>Path: {filepath}</p>\n'
        
        html += f'    </div>\n'
        
        # Add metadata
        html += f'    <div class="metadata">\n'
        
        # First, show common metadata for this media
        html += f'        <h3>Media Information</h3>\n'
        html += f'        <table>\n'
        html += f'            <tr><th>Property</th><th>Value</th></tr>\n'
        
        # Always include local filepath
        html += f'            <tr><td>local_filepath</td><td>{media_item["local_filepath"]}</td></tr>\n'
        
        for col in metadata_cols:
            if col in media_item and col != 'local_filepath':  # Skip filepath as we already added it
                value = media_item[col]
                
                # Format timestamp if present
                if col == 'timestamp' and isinstance(value, (int, float)):
                    try:
                        value = datetime.fromtimestamp(value).strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        pass  # Keep original if conversion fails
                
                # Handle different data types for display
                if isinstance(value, (list, dict)):
                    value = str(value)
                
                html += f'            <tr><td>{col}</td><td>{value}</td></tr>\n'
        
        html += f'        </table>\n'
        
        # Then, show all predictions for this media
        html += f'        <h3>Predictions ({len(media_item["predictions"])})</h3>\n'
        html += f'        <table class="predictions-table">\n'
        html += f'            <tr><th>Miner UID</th><th>Prediction</th><th>Label</th><th>Binary Correct</th><th>Multiclass Correct</th></tr>\n'
        
        for pred in media_item['predictions']:
            uid = pred.get('uid', 'Unknown')
            prediction = pred.get('prediction', 'N/A')
            label = pred.get('label', 'N/A')
            
            # Format prediction to bold the largest value if it's a list or similar
            prediction_display = prediction
            if isinstance(prediction, (list, tuple, np.ndarray)) or (isinstance(prediction, str) and '[' in prediction and ']' in prediction):
                # Convert string representation of list to actual list if needed
                if isinstance(prediction, str):
                    try:
                        # Try to convert "[0.1, 0.2, 0.7]" to a list
                        import ast
                        prediction_values = ast.literal_eval(prediction)
                    except:
                        prediction_values = prediction
                else:
                    prediction_values = prediction
                
                # If it's now a list-like object, format it with the max value in bold
                if isinstance(prediction_values, (list, tuple, np.ndarray)):
                    if len(prediction_values) > 0:
                        max_index = np.argmax(prediction_values)
                        prediction_str = str(prediction_values)
                        # Convert "[0.1, 0.2, 0.7]" to "[0.1, 0.2, <b>0.7</b>]"
                        parts = prediction_str.split(str(prediction_values[max_index]))
                        if len(parts) >= 2:
                            prediction_display = parts[0] + f"<b>{prediction_values[max_index]}</b>" + parts[1]
                        else:
                            prediction_display = prediction_str
                    else:
                        prediction_display = str(prediction_values)
                else:
                    prediction_display = str(prediction)
            
            # Check multiclass correctness (exact match)
            multiclass_correct = np.argmax(prediction) == label
            
            # Check binary correctness (both 1 or 2, or both something else)
            binary_correct = False
            if label in [1, 2] and np.argmax(prediction) in [1, 2]:
                binary_correct = True
            elif label not in [1, 2] and np.argmax(prediction) not in [1, 2]:
                binary_correct = True
            
            # Determine row class based on correctness
            if binary_correct and multiclass_correct:
                row_class = 'correct-both'  # Green
                binary_icon = "✓"
                multiclass_icon = "✓"
            elif binary_correct or multiclass_correct:
                row_class = 'correct-one'   # Yellow
                binary_icon = "✓" if binary_correct else "✗"
                multiclass_icon = "✓" if multiclass_correct else "✗"
            else:
                row_class = 'correct-none'  # Red
                binary_icon = "✗"
                multiclass_icon = "✗"
            
            html += f'            <tr class="{row_class}"><td>{uid}</td><td>{prediction_display}</td><td>{label}</td><td>{binary_icon}</td><td>{multiclass_icon}</td></tr>\n'
        
        html += f'        </table>\n'
        html += f'    </div>\n'
        html += f'</div>\n'
    
    # Close HTML
    html += """
        </div>
    </body>
    </html>
    """
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Write HTML to file
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"Gallery generated successfully at: {output_path}")
    return output_path
