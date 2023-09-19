import argparse

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.worker.cache_engine import CacheEngine


def main(args: argparse.Namespace):
    # Parse the CLI argument and initialize the engine.
    engine_args = EngineArgs.from_cli_args(args)
    engine_configs = engine_args.create_engine_configs()
    cache_block_size = CacheEngine.get_cache_block_size(
            engine_configs[1].block_size, engine_configs[0], engine_configs[2])
    print(cache_block_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)