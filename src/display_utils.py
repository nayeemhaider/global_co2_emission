def rename_for_display(df):
    return df.rename(columns={
        "emission": "emission (tons)",
        "population2022": "population (people)",
        "area": "area (km²)",
        "density_km2": "density (people/km²)",
        "share_of_world": "share_of_world (fraction of world)"
    })
