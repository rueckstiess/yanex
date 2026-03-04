"""Tests for sync target parsing."""

import pytest

from yanex.sync.target import (
    DEFAULT_SSH_PATH,
    SyncProtocol,
    SyncTarget,
    parse_target,
)


class TestParseTarget:
    """Tests for parse_target()."""

    def test_s3_bucket_only(self):
        target = parse_target("s3://my-bucket")
        assert target.protocol == SyncProtocol.S3
        assert target.bucket == "my-bucket"
        assert target.prefix is None

    def test_s3_bucket_with_prefix(self):
        target = parse_target("s3://my-bucket/experiments")
        assert target.protocol == SyncProtocol.S3
        assert target.bucket == "my-bucket"
        assert target.prefix == "experiments"

    def test_s3_bucket_with_nested_prefix(self):
        target = parse_target("s3://my-bucket/path/to/experiments")
        assert target.protocol == SyncProtocol.S3
        assert target.bucket == "my-bucket"
        assert target.prefix == "path/to/experiments"

    def test_s3_trailing_slash_stripped(self):
        target = parse_target("s3://my-bucket/experiments/")
        assert target.prefix == "experiments"

    def test_s3_empty_bucket_raises(self):
        with pytest.raises(ValueError, match="bucket name"):
            parse_target("s3://")

    def test_s3_empty_after_scheme_raises(self):
        with pytest.raises(ValueError, match="bucket name"):
            parse_target("s3:///prefix")

    def test_ssh_host_only(self):
        target = parse_target("sky-dev")
        assert target.protocol == SyncProtocol.SSH
        assert target.host == "sky-dev"
        assert target.user is None
        assert target.path == DEFAULT_SSH_PATH

    def test_ssh_host_with_path(self):
        target = parse_target("sky-dev:~/custom/path")
        assert target.protocol == SyncProtocol.SSH
        assert target.host == "sky-dev"
        assert target.path == "~/custom/path/"

    def test_ssh_host_with_path_trailing_slash(self):
        target = parse_target("sky-dev:~/custom/path/")
        assert target.path == "~/custom/path/"

    def test_ssh_user_host(self):
        target = parse_target("user@gpu-box")
        assert target.host == "gpu-box"
        assert target.user == "user"
        assert target.path == DEFAULT_SSH_PATH

    def test_ssh_user_host_path(self):
        target = parse_target("user@host:/data/experiments")
        assert target.host == "host"
        assert target.user == "user"
        assert target.path == "/data/experiments/"

    def test_ssh_host_with_colon_empty_path(self):
        target = parse_target("sky-dev:")
        assert target.host == "sky-dev"
        assert target.path == DEFAULT_SSH_PATH

    def test_ssh_ip_address(self):
        target = parse_target("192.168.1.100")
        assert target.host == "192.168.1.100"
        assert target.user is None

    def test_ssh_user_ip_path(self):
        target = parse_target("root@10.0.0.1:/opt/yanex/experiments")
        assert target.host == "10.0.0.1"
        assert target.user == "root"
        assert target.path == "/opt/yanex/experiments/"

    def test_empty_target_raises(self):
        with pytest.raises(ValueError, match="empty"):
            parse_target("")

    def test_whitespace_target_raises(self):
        with pytest.raises(ValueError, match="empty"):
            parse_target("   ")

    def test_empty_user_raises(self):
        with pytest.raises(ValueError, match="empty username"):
            parse_target("@host")

    def test_empty_host_raises(self):
        with pytest.raises(ValueError, match="hostname"):
            parse_target("user@")

    def test_target_stripped(self):
        target = parse_target("  sky-dev  ")
        assert target.host == "sky-dev"


class TestSyncTargetProperties:
    """Tests for SyncTarget computed properties."""

    def test_ssh_dest_host_only(self):
        target = SyncTarget(
            protocol=SyncProtocol.SSH, host="sky-dev", path="~/.yanex/experiments/"
        )
        assert target.ssh_dest == "sky-dev:~/.yanex/experiments/"

    def test_ssh_dest_with_user(self):
        target = SyncTarget(
            protocol=SyncProtocol.SSH, host="gpu-box", user="root", path="/data/"
        )
        assert target.ssh_dest == "root@gpu-box:/data/"

    def test_ssh_dest_raises_for_s3(self):
        target = SyncTarget(protocol=SyncProtocol.S3, bucket="b")
        with pytest.raises(ValueError, match="SSH"):
            _ = target.ssh_dest

    def test_s3_url_bucket_only(self):
        target = SyncTarget(protocol=SyncProtocol.S3, bucket="my-bucket")
        assert target.s3_url == "s3://my-bucket"

    def test_s3_url_with_prefix(self):
        target = SyncTarget(protocol=SyncProtocol.S3, bucket="my-bucket", prefix="exp")
        assert target.s3_url == "s3://my-bucket/exp"

    def test_s3_url_raises_for_ssh(self):
        target = SyncTarget(protocol=SyncProtocol.SSH, host="h", path="/")
        with pytest.raises(ValueError, match="S3"):
            _ = target.s3_url

    def test_display_name_ssh(self):
        target = SyncTarget(
            protocol=SyncProtocol.SSH, host="sky-dev", path="~/.yanex/experiments/"
        )
        assert target.display_name == "sky-dev:~/.yanex/experiments/"

    def test_display_name_s3(self):
        target = SyncTarget(protocol=SyncProtocol.S3, bucket="b", prefix="p")
        assert target.display_name == "s3://b/p"
