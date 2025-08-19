"""
Configuration manager for loading YAML configs
"""

import yaml
import argparse
from typing import Dict, Any


class ConfigManager:
    """Manage configuration loading and merging"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
        """Merge config file with command line arguments
        Priority: defaults < config < explicit CLI args
        """
        # Create a new namespace starting with parser defaults
        merged_args = argparse.Namespace()
        
        # Step 1: Set parser defaults
        for action in parser._actions:
            if action.dest != 'help' and hasattr(action, 'default'):
                setattr(merged_args, action.dest, action.default)
        
        # Step 2: Override with config values
        for key, value in config.items():
            if hasattr(merged_args, key):  # Only set if it's a valid argument
                setattr(merged_args, key, value)
        
        # Step 3: Override with explicitly provided CLI arguments
        # Parse sys.argv to find which args were explicitly provided
        import sys
        provided_args = set()
        i = 1
        while i < len(sys.argv):
            if sys.argv[i].startswith('--'):
                arg_name = sys.argv[i][2:].replace('-', '_')
                provided_args.add(arg_name)
                # Skip the value if it exists and is not another flag
                if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('--'):
                    i += 1
            i += 1
        
        # Only override config with explicitly provided CLI args
        cli_dict = vars(args)
        for key, value in cli_dict.items():
            if key in provided_args:
                setattr(merged_args, key, value)
        
        return merged_args