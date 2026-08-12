# LinkedIn

Source: `docs/tutorials/51-tcp-quarantine/README.md`

## Post

Firewalls can block new traffic, but they cannot tear down one already-established TCP connection without disrupting the rest. This tutorial uses a BPF TCP iterator and bpf_sock_destroy to identify and terminate exactly one IPv4 4-tuple from the kernel.

https://eunomia.dev/tutorials/51-tcp-quarantine/

## Publishing

- First visible lines: the established-connection problem and exact 4-tuple result are visible before "see more."
- Primary link: https://eunomia.dev/tutorials/51-tcp-quarantine/
- Media: LinkedIn link preview card only.
- Visibility: Anyone.
- Published URL: https://www.linkedin.com/feed/update/urn:li:share:7493164412321812480/
- QA state: Confirmed on the public post. The exact two-sentence body, public visibility, LinkedIn short-link destination, and tutorial preview card with its title, image, and eunomia.dev domain rendered correctly.
