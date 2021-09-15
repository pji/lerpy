#! .venv/bin/python
"""
resize_image
~~~~~~~~~~~~

Resize an image.
"""
from argparse import ArgumentParser

import imgwriter as iw

import lerpy as lp


# Constants.
INTERPOLATIONS = {
    'bilinear': lp.ndlerp,
    'bicubic': lp.ndcerp,
}


# CLI handling.
def get_cli_args() -> None:
    """Parse the command line instruction."""
    # Configuration for the command line options.
    options = (
        {
            'args': ('srcfile', ),
            'kwargs': {
                'type': str,
                'action': 'store',
                'help': 'The path to the file to resize.'
            },
        },
        {
            'args': ('dstfile', ),
            'kwargs': {
                'type': str,
                'action': 'store',
                'help': 'The path for the resized file.'
            },
        },
        {
            'args': ('-i', '--interpolation', ),
            'kwargs': {
                'type': str,
                'choices': INTERPOLATIONS,
                'default': next(iter(INTERPOLATIONS)),
                'action': 'store',
                'help': 'The interpolation method for the resize.'
            },
        },
        {
            'args': ('-m', '--magnify', ),
            'kwargs': {
                'type': float,
                'action': 'store',
                'help': 'The magnification factor for the resize.'
            },
        },
        {
            'args': ('-s', '--size', ),
            'kwargs': {
                'type': int,
                'nargs': 2,
                'action': 'store',
                'help': 'The dimensions to resize to.'
            },
        },
    )

    # Build and configure the argument parser.
    p = ArgumentParser(**{
        'prog': 'resize_image.py',
        'description': 'Resize an image with interpolation.',
    })
    for option in options:
        p.add_argument(*option['args'], **option['kwargs'])

    # Return the parsed arguments.
    return p.parse_args()


# Mainline.
def main() -> None:
    """The mainline for the script."""
    # Get the arguments from the command line.
    args = get_cli_args()

    # Read in the image to resize.
    src = iw.read_image(args.srcfile)

    # Determine which interpolation function to use for the resize.
    erp = INTERPOLATIONS[args.interpolation]
    
    # You have to be either magnifying or resizing.
    if not args.magnify and not args.size:
        msg = 'Must specify either a magnification factor or new size.'
        raise ValueError(msg)
    
    # You can't both magnify and resize.
    elif args.magnify and args.size:
        msg = 'Must only specify a magnification factor or new side.'

    # Calculate the final size if we are magnifying.
    elif args.magnify and not args.size:
        src_dims = src.shape
        if len(src.shape) > 2:
            src_dims = src_dims[1:3]
        dst_dims = lp.magnify_size(src_dims, args.magnify)
        if len(src.shape) > 2:
            dst_shape = (src.shape[0], *dst_dims)
        if len(src.shape) == 4:
            dst_shape = (*dst_shape, src.shape[-1])

    # Handle extra dimensions and the color channels when resizing.
    elif not args.magnify and args.size:
        if len(src.shape) == 4:
            dst_shape = (src.shape[0], *args.size, src.shape[-1])
        elif len(src.shape) == 3:
            dst_shape = (src.shape[0], *args.size)
        else:
            dst_shape = tuple(args.size)

    # Perform the resizing and save the result.
    dst = lp.resize_array(src, dst_shape, erp)
    iw.save(args.dstfile, dst)


if __name__ == '__main__':
    main()
