import argparse

def main():
    parser = argparse.ArgumentParser(description="CLI for the cellSAM package.")
    subparsers = parser.add_subparsers(help='sub-command help')

    # Example of adding a sub-parser for the napari plugin
    parser_napari = subparsers.add_parser('napari', help='Run the napari plugin')
    parser_napari.set_defaults(func=run_napari)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func()
    else:
        parser.print_help()

def run_napari():
    from cellSAM.napari_plugin._widget import CellSAMWidget
    import napari
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(CellSAMWidget(viewer))
    napari.run()


if __name__ == "__main__":
    main()
