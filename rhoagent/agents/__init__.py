# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from rhoagent.agent import Agent, BasicAgent
from rhoagent.multi_agent_hub import MultiAgentHub

from .assistant import Assistant
# DocQAAgent is the default solution for long document question answering.
# The actual implementation of DocQAAgent may change with every release.
from .doc_qa import BasicDocQA as DocQAAgent
from .doc_qa import ParallelDocQA
from .fncall_agent import FnCallAgent
from .user_agent import UserAgent

__all__ = [
    'Agent',
    'BasicAgent',
    'MultiAgentHub',
    'DocQAAgent',
    'ParallelDocQA',
    'Assistant',
    'UserAgent',
    'FnCallAgent',
]
