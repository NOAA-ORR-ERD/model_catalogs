#!/usr/bin/env pytest
# -*- coding: utf-8 -*-
"""Unit tests for example support module."""
from model_catalogs.examples import parse_bbox


def test_bbox_parse():
    """Test that bbox can be parsed."""
    bbox = parse_bbox("1,2,3,4")
    assert bbox == (1.0, 2.0, 3.0, 4.0)
