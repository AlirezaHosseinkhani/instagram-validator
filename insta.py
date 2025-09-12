import instaloader
from urllib.parse import urlparse


def shortcode_from_url(url: str):
    path = urlparse(url).path.strip("/")
    parts = path.split("/")
    if len(parts) >= 2 and parts[0] in ("p", "reel", "tv"):
        return parts[1], parts[0]  # return shortcode and type
    return None, None


def get_post_info(post_url: str, user: str = None, password: str = None):
    shortcode, ctype = shortcode_from_url(post_url)
    if not shortcode:
        return None

    L = instaloader.Instaloader()
    if user and password:  # login if you want private account access
        L.login(user, password)

    post = instaloader.Post.from_shortcode(L.context, shortcode)
    profile = post.owner_profile

    content_type = {
        "p": "post",
        "reel": "reel",
        "tv": "igtv"
    }.get(ctype, "unknown")

    return {
        "username": post.owner_username,
        "is_private": profile.is_private,
        "is_verified": profile.is_verified,
        "content_type": content_type,
        "caption": post.caption,
        "hashtags": list(post.caption_hashtags),
        "mentions": list(post.caption_mentions),
    }


# Example usage
url = "https://www.instagram.com/reel/DBo1NGICI-z/"
info = get_post_info(url)
print("Username:", info["username"])
print("Private account?:", info["is_private"])
print("Verified?:", info["is_verified"])
print("Content type:", info["content_type"])
print("Hashtags:", info["hashtags"])
