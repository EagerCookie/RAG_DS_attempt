@router.get("/loaders")
def list_loaders():
    return registry.get_all_loaders()

@router.get("/loaders/{id}/config")
def get_config(id: str):
    return registry.get_loader_config(id)
