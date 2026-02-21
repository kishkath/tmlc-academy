# import os
#
# project_name = "qwen_finetune_project"
# os.makedirs(project_name, exist_ok=True)
#
# files_content = {
#     "config.json": """<PASTE config.json CONTENT HERE>""",
#     "test_config.json": """<PASTE test_config.json CONTENT HERE>""",
#     "data_utils.py": """<PASTE data_utils.py CONTENT HERE>""",
#     "model_utils.py": """<PASTE model_utils.py CONTENT HERE>""",
#     "trainer_utils.py": """<PASTE trainer_utils.py CONTENT HERE>""",
#     "train.py": """<PASTE train.py CONTENT HERE>""",
#     "save_inference.py": """<PASTE save_inference.py CONTENT HERE>""",
#     "README.md": """<PASTE README.md CONTENT HERE>"""
# }
#
# for fname, content in files_content.items():
#     with open(os.path.join(project_name, fname), "w") as f:
#         f.write(content)
#
# print(f"Project folder '{project_name}' created successfully!")
#
# Replace <PASTE ... CONTENT HERE> with the exact contents of each file from above. After running, zip the folder qwen_finetune_project and you have a ready-to-use package.
#
# I can also provide a ready-to-run zipped base64 string that you can decode to get all files, if you want to avoid manually pasting.