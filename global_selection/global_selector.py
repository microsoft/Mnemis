import os
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
load_dotenv()
import asyncio
import json
from time import time
from typing import Literal
from neo4j import AsyncDriver, AsyncGraphDatabase
from async_lru import alru_cache

from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI
from pydantic import BaseModel

from graphiti_core.llm_client import LLMClient, ModelSize, OpenAIClient
from graphiti_core.prompts.models import Message
from .prompts import NODE_SELECTION_PROMPT_TEMPLATE

CACHE_SIZE_PER_QUERY = 500

class NodeSelection(BaseModel):
    name: str
    uuid: str
    get_all_children: bool

class NodeSelectionList(BaseModel):
    selections: list[NodeSelection]

class Query:
    GET_MAX_LAYER = """
    match (n:Category) where n.group_id = $group_id
    return max(n.layer) as max_layer
    """
    
    GET_NODES_BY_LAYER = """
    match (n:{label}) where n.group_id = $group_id
    return n.uuid as uuid, n.name as name, n.tag as tag, n.summary as summary
    """
    
    GET_CHILD_NODES = """
    match (parent:Category)-[:CATEGORIZES]->(child:Category|Entity) where parent.group_id = $group_id and parent.uuid = $parent_uuid
    return distinct child.uuid as uuid, child.name as name, child.tag as tag, child.summary as summary, child.layer as layer
    """
    
    GET_CHILD_NODES_BATCH = """
    match (parent:Category)-[:CATEGORIZES]->(child:Category|Entity) where parent.group_id = $group_id and parent.uuid in $parent_uuids
    return distinct child.uuid as uuid, child.name as name, child.tag as tag, child.summary as summary, child.layer as layer
    """
    
    GET_ALL_DESCENDANTS = """
    match (parent:Category|Entity)-[:CATEGORIZES*1..]->(child:Category|Entity) where parent.group_id = $group_id and parent.uuid = $parent_uuid
    return distinct child.uuid as uuid, child.name as name, child.tag as tag, child.summary as summary
    """
    
    GET_ALL_DESCENDANTS_BATCH = """
    match (parent:Category|Entity)-[:CATEGORIZES*1..]->(child:Category|Entity) where parent.group_id = $group_id and parent.uuid in $parent_uuids
    return distinct child.uuid as uuid, child.name as name, child.tag as tag, child.summary as summary
    """
    
    GET_ONE_HOP_EPISODES = """
    match (n)-[:MENTIONS]-(m:Episodic)
    where n.group_id = $group_id and n.uuid = $node_uuid
    return distinct m.uuid as uuid, m.content as content, m.valid_at as valid_at
    """
    
    GET_ONE_HOP_EPISODES_BATCH = """
    match (n)-[:MENTIONS]-(m:Episodic)
    where n.group_id = $group_id and n.uuid in $node_uuids
    return distinct m.uuid as uuid, m.content as content, m.valid_at as valid_at
    """

    GET_ONE_HOP_NODES_AND_EDGES = """
    match (n)-[r:RELATES_TO]-(m:Entity)
    where n.group_id = $group_id and n.uuid = $node_uuid
    return r.uuid as fact_uuid, r.fact as fact, r.valid_at as valid_at, r.invalid_at as invalid_at, m.uuid as entity_uuid, m.name as name, m.tag as tag, m.summary as summary
    """
    
    GET_ONE_HOP_NODES_AND_EDGES_BATCH = """
    match (n)-[r:RELATES_TO]-(m:Entity)
    where n.group_id = $group_id and n.uuid in $node_uuids
    return r.uuid as fact_uuid, r.fact as fact, r.valid_at as valid_at, r.invalid_at as invalid_at, m.uuid as entity_uuid, m.name as name, m.tag as tag, m.summary as summary
    """

class GlobalSelectorConfig(BaseModel):
    use_summary: bool = False
    use_tag: bool = True


class GlobalSelector:
    def __init__(self, driver: AsyncDriver, llm_client: LLMClient, selection_config: GlobalSelectorConfig = GlobalSelectorConfig()):
        self.driver = driver
        self.llm_client = llm_client
        self.selection_config = selection_config

    def clear_cache(self):
        self.get_max_layer.cache_clear()
        self.get_nodes_by_layer.cache_clear()
        self.get_child_nodes.cache_clear()
        self.get_all_descendants.cache_clear()
        self.get_one_hop_neighbors.cache_clear()
    
    @alru_cache(maxsize=CACHE_SIZE_PER_QUERY)
    async def get_max_layer(self, group_id: str) -> int:
        result = await self.driver.execute_query(
            Query.GET_MAX_LAYER,
            group_id=group_id
        )
        record = result.records[0]
        return record['max_layer'] if record['max_layer'] is not None else 0
    
    @alru_cache(maxsize=CACHE_SIZE_PER_QUERY)
    async def get_nodes_by_layer(self, layer: int, group_id: str) -> list[dict]:
        result = await self.driver.execute_query(
            Query.GET_NODES_BY_LAYER.format(label=f'Category_{layer}' if layer > 0 else 'Entity'),
            group_id=group_id
        )
        return [dict(record) for record in result.records]

    @alru_cache(maxsize=CACHE_SIZE_PER_QUERY)
    async def get_child_nodes(self, parent_uuid: str, group_id: str) -> list[dict]:
        result = await self.driver.execute_query(
            Query.GET_CHILD_NODES,
            group_id=group_id,
            parent_uuid=parent_uuid
        )
        return [dict(record) for record in result.records]

    async def get_child_nodes_batch(self, parent_uuids: list[str], group_id: str, mode: Literal['mp', 'batch'] = 'mp') -> list[dict]:
        if mode == 'mp':
            tasks = [self.get_child_nodes(uuid, group_id=group_id) for uuid in parent_uuids]
            results = await asyncio.gather(*tasks)
            return list({item['uuid']: item for sublist in results for item in sublist}.values())
        elif mode == 'batch':
            result = await self.driver.execute_query(
                Query.GET_CHILD_NODES_BATCH,
                group_id=group_id,
                parent_uuids=parent_uuids
            )
            return [dict(record) for record in result.records]
        else:
            raise ValueError("mode must be 'mp' or 'batch'")
    
    @alru_cache(maxsize=CACHE_SIZE_PER_QUERY)
    async def get_all_descendants(self, parent_uuid: str, group_id: str) -> list[dict]:
        result = await self.driver.execute_query(
            Query.GET_ALL_DESCENDANTS,
            group_id=group_id,
            parent_uuid=parent_uuid
        )
        return [dict(record) for record in result.records]

    async def get_all_descendants_batch(self, parent_uuids: list[str], group_id: str, mode: Literal['mp', 'batch'] = 'mp') -> list[dict]:
        if mode == 'mp':
            tasks = [self.get_all_descendants(parent_uuid, group_id=group_id) for parent_uuid in parent_uuids]
            results = await asyncio.gather(*tasks)
            return list({item['uuid']: item for sublist in results for item in sublist}.values())
        elif mode == 'batch':
            result = await self.driver.execute_query(
                Query.GET_ALL_DESCENDANTS_BATCH,
                group_id=group_id,
                parent_uuids=parent_uuids
            )
            return [dict(record) for record in result.records]
        else:
            raise ValueError("mode must be 'mp' or 'batch'")
    
    def _gather_neighbors(self, results: list) -> dict:
        assert len(results) == 2
        episodes = [dict(record) for record in results[0].records]
        edges = {}
        nodes = {}
        for record in results[1].records:
            neighbor = dict(record)
            
            edges[neighbor['fact_uuid']] = {
                'fact': neighbor['fact'],
                'valid_at': neighbor['valid_at'],
                'invalid_at': neighbor['invalid_at'],
                'uuid': neighbor['fact_uuid']
            }
            
            nodes[neighbor['entity_uuid']] = {
                'uuid': neighbor['entity_uuid'],
                'name': neighbor['name'],
                'tag': neighbor['tag'],
                'summary': neighbor['summary']
            }

        return {
            'episodes': episodes,
            'edges': list(edges.values()),
            'nodes': list(nodes.values())
        }
    
    @alru_cache(maxsize=CACHE_SIZE_PER_QUERY)
    async def get_one_hop_neighbors(self, node_uuid: str, group_id: str) -> dict:
        tasks = [self.driver.execute_query(query, group_id=group_id, node_uuid=node_uuid) for query in [Query.GET_ONE_HOP_EPISODES, Query.GET_ONE_HOP_NODES_AND_EDGES]]

        ep_result, result = await asyncio.gather(*tasks)
        return self._gather_neighbors([ep_result, result])

    async def get_one_hop_neighbors_batch(self, node_uuids: list[str], group_id: str, mode: Literal['mp', 'batch'] = 'mp') -> dict:
        if mode == 'mp':
            tasks = [self.get_one_hop_neighbors(uuid, group_id=group_id) for uuid in node_uuids]
            results = await asyncio.gather(*tasks)

            episodes = list({item['uuid']: item for res in results for item in res['episodes']}.values())
            edges = list({item['uuid']: item for res in results for item in res['edges']}.values())
            nodes = list({item['uuid']: item for res in results for item in res['nodes']}.values())

            return {
                'episodes': episodes,
                'edges': edges,
                'nodes': nodes
            }
        elif mode == 'batch':
            tasks = [self.driver.execute_query(query, group_id=group_id, node_uuids=node_uuids) for query in [Query.GET_ONE_HOP_EPISODES_BATCH, Query.GET_ONE_HOP_NODES_AND_EDGES_BATCH]]
            results = await asyncio.gather(*tasks)
            return self._gather_neighbors(results)
        else:
            raise ValueError("mode must be 'mp' or 'batch'")

    async def layer_selection(self, query: str, current_layer_categories: list[dict]) -> tuple[list[dict], list[dict]]:
        allowed_fields = [
            "uuid",
            "name",
            *(['tag'] if self.selection_config.use_tag else []),
            *(['summary'] if self.selection_config.use_summary else []),
        ]
        category_dict = {cat['uuid']: cat for cat in current_layer_categories}
        category_name_dict = {cat['name']: cat for cat in current_layer_categories}
        category_context = [{
            field: cat[field] for field in allowed_fields
        } for cat in current_layer_categories]
        
        prompt = NODE_SELECTION_PROMPT_TEMPLATE.format(
            query=query,
            nodes_info='\n'.join([json.dumps(cat, ensure_ascii=False) for cat in category_context])
        )
        response = await self.llm_client.generate_response(
            messages=[Message(role='user', content=prompt)],
            response_model=NodeSelectionList,
            model_size=ModelSize.large
        )

        selected_categories = []
        shortcut_categories = []
        for selection in response.get('selections', []):
            name = selection['name']
            uuid = selection['uuid']
            get_all_children = selection['get_all_children']
            
            if uuid in category_dict:
                selected_nodes = category_dict[uuid]
            elif name in category_name_dict:
                selected_nodes = category_name_dict[name]
            else:
                print(f"Warning: LLM returned name '{name}' and uuid '{uuid}' that do not match any input node. Skipping.")
                continue
            if get_all_children:
                shortcut_categories.append(selected_nodes)
            else:
                selected_categories.append(selected_nodes)
        return selected_categories, shortcut_categories

    async def global_selection(self, query: str, group_id: str) -> tuple[dict, dict]:
        start = time()
        mode = 'batch'
        time_stats = {}
        max_layer = await self.get_max_layer(group_id=group_id)
        time_stats['init'] = time() - start

        start = time()
        previous_layer_categories = []
        selected_categories = {}
        for layer in range(max_layer, 0, -1):
            if layer == max_layer:
                current_layer_categories = await self.get_nodes_by_layer(layer, group_id=group_id)
            elif len(previous_layer_categories) > 0:
                current_layer_categories = await self.get_child_nodes_batch([cat['uuid'] for cat in previous_layer_categories], group_id=group_id, mode=mode)
            else:
                break

            selected, shortcuts = await self.layer_selection(query, current_layer_categories)
            
            for cat in selected:
                selected_categories[cat['uuid']] = cat
            all_descendants = await self.get_all_descendants_batch([cat['uuid'] for cat in shortcuts], group_id=group_id, mode=mode)
            for cat in all_descendants:
                selected_categories[cat['uuid']] = cat
            
            # print('Current Layer:', layer, 'Current Nodes:', len(current_layer_categories), 'Selected:', len(selected), 'Shortcuts:', len(shortcuts), 'All Selected:', len(selected_categories))
            previous_layer_categories = selected

        time_stats['layer_selection'] = time() - start
        
        start = time()
        neighbors = await self.get_one_hop_neighbors_batch([cat['uuid'] for cat in selected_categories.values()], group_id=group_id, mode=mode)
        time_stats['one_hop_neighbors'] = time() - start

        def format(item: dict):
            if item.get('valid_at') and type(item['valid_at']) != str:
                    item['valid_at'] = item['valid_at'].strftime("%Y/%m/%d (%a) %H:%M")
            if item.get('invalid_at') and type(item['invalid_at']) != str:
                item['invalid_at'] = item['invalid_at'].strftime("%Y/%m/%d (%a) %H:%M")
            return item
        
        episodes = [format(ep) for ep in neighbors['episodes']]
        edges = [format(edge) for edge in neighbors['edges']]
        selected_categories.update({ent['uuid']: ent for ent in neighbors['nodes']})
        nodes = [format(node) for node in selected_categories.values()]
        
        results = {
            'episodes': episodes,
            'edges': edges,
            'nodes': nodes,
        }
        return results, time_stats

def load_locomo_data_query_group_id(file_path, group_id_prefix='locomo_ziyu'):
    """
    Load the locomo data from a JSON file.
    
    Args:
        file_path (str): The path to the JSON file containing locomo data.
        
    Returns:
        dict: The locomo data as a dictionary.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    query_groupid_list = []
    
    for user_id, user_data in enumerate(data):
        for query_data in user_data['qa']:
            query_groupid_list.append({
                'group_id': f"{group_id_prefix}_{user_id}",
                'query': query_data['question']
            })
    print(f"Loaded {len(query_groupid_list)} queries from locomo data.")
    return query_groupid_list

def load_lme_data_query_group_id(file_path, group_id_prefix='lme_s_ziyu'):
    """
    Load the lme data from a JSON file.
    
    Args:
        file_path (str): The path to the JSON file containing lme data.
        group_id_prefix (str): The prefix to use for the group IDs.
    Returns:
        dict: The lme data as a dictionary.
    """
    data = pd.read_json(file_path).to_dict(orient='records')
    
    query_groupid_list = []
    for user_id, user_data in enumerate(data):
        query_groupid_list.append({
            'group_id': f"{group_id_prefix}_{user_id}",
            'query': user_data['question'],
            'question_id': user_data['question_id']
        })     
    print(f"Loaded {len(query_groupid_list)} queries from lme data.")
    return query_groupid_list

async def get_global_search_context(query_groupid_list, global_searcher: GlobalSelector, max_concurrent=5):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def search_with_semaphore(query_data):
        async with semaphore:
            group_id = query_data['group_id']
            query = query_data['query']
            context, time_stats = await global_searcher.global_selection(query, group_id=group_id)
            return {
                "query": query,
                "group_id": group_id,
                "context": context,
                "time_stats": time_stats
            }
    # Create tasks for all queries
    tasks = [search_with_semaphore(query_data) for query_data in query_groupid_list]

    # Execute all tasks concurrently with semaphore control
    search_results = await asyncio.gather(*tasks)
    return search_results

async def parse_locomo(selector: GlobalSelector):
    group_id_prefix = 'locomo_mnemis_coreAI_tel_b20_nec_full'
    max_concurrent = 10
    batch_size = 50
    output_path = '/data/zh/gs/v2_locomo_mnemis_coreAI_tel_b20_nec_full.json'

    query_groupid_list = load_locomo_data_query_group_id('data/locomo.json', group_id_prefix=group_id_prefix)
    all_data_count = len(query_groupid_list)
    
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for i in tqdm(range(0, all_data_count, batch_size), desc=f"Processing Batches (locomo, b={batch_size})"):
            batch = query_groupid_list[i:i + batch_size]
            
            # Run the global search context retrieval
            search_results = await get_global_search_context(batch, selector, max_concurrent=max_concurrent)
            
            # Save the results to the output file
            for result in search_results:
                print(json.dumps(result, ensure_ascii=False), file=output_file)
            print(f"Processed {len(batch)} queries, saving results...")
            print(f"Batch {i // batch_size + 1} results saved to {output_path}")

async def parse_lme(selector: GlobalSelector):
    group_id_prefix = 'lme_s_mnemis_coreAI_tel_b20_nec_full'
    max_concurrent = 10
    batch_size = 30
    output_path = '/data/zh/gs/v2_lme_s_mnemis_coreAI_tel_b20_nec_full.json'

    query_groupid_list = load_lme_data_query_group_id('data/longmemeval_s.json', group_id_prefix=group_id_prefix)
    all_data_count = len(query_groupid_list)
    
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for i in tqdm(range(0, all_data_count, batch_size), desc=f"Processing Batches (lme_s, b={batch_size})"):
            batch = query_groupid_list[i:i + batch_size]
            
            # Run the global search context retrieval
            search_results = await get_global_search_context(batch, selector, max_concurrent=max_concurrent)
            
            # Save the results to the output file
            for result in search_results:
                print(json.dumps(result, ensure_ascii=False), file=output_file)
            print(f"Processed {len(batch)} queries, saving results...")
            print(f"Batch {i // batch_size + 1} results saved to {output_path}")

async def main():
    url = 'bolt://localhost:7687'
    user = 'xxx'
    password = 'xxx'
    driver = AsyncGraphDatabase.driver(url, auth=(user, password), max_connection_pool_size=1000)
    raw_client = AsyncOpenAI(
        base_url="xxx",
        api_key="EMPTY"
    )
    llm_client = OpenAIClient(client=raw_client)
    selector = GlobalSelector(driver, llm_client)
    
    start = time()
    await parse_locomo(selector)
    # await parse_lme(selector)
    end = time()
    print(f"Total time taken: {end - start} s")
    print(selector.llm_client.get_token_stats())


if __name__ == "__main__":
    asyncio.run(main())
