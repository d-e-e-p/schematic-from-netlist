from typing import List, Tuple


class Geom:
    @staticmethod
    def sgn(x: int) -> int:
        return 0 if x == 0 else (1 if x > 0 else -1)

    @staticmethod
    def is_adjacent(p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
        """Check if two points are adjacent on the grid (Manhattan distance = 1)."""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) == 1

    @staticmethod
    def is_turn(p1: Tuple[int, int], p2: Tuple[int, int], p3: Tuple[int, int]) -> bool:
        """Return True if p2 is a turning point (non-collinear with p1->p3)."""
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        return v1[0] * v2[1] - v1[1] * v2[0] != 0

    @staticmethod
    def merge_collinear_segments(segments: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Merge consecutive collinear segments into maximal straight runs.
        Input: segments = [((x1,y1),(x2,y2)), ...] where segments are contiguous.
        """
        if not segments:
            return []

        cur_s, cur_e = segments[0]
        prev_dir = (Geom.sgn(cur_e[0] - cur_s[0]), Geom.sgn(cur_e[1] - cur_s[1]))
        merged = []

        for s, e in segments[1:]:
            # if not contiguous, flush and start new
            if s != cur_e:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
                prev_dir = (Geom.sgn(cur_e[0] - cur_s[0]), Geom.sgn(cur_e[1] - cur_s[1]))
                continue

            dir_ = (Geom.sgn(e[0] - s[0]), Geom.sgn(e[1] - s[1]))
            if dir_ == prev_dir:
                # same direction → extend current segment
                cur_e = e
            else:
                # direction changed → close current and start new
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
                prev_dir = dir_

        merged.append((cur_s, cur_e))
        return merged

    @staticmethod
    def process_path(path: List[Tuple[int, int, int]], stop: bool) -> Tuple[List[Tuple[Tuple[int, int], Tuple[int, int]]], List[Tuple[int, int]]]:
        """
        Convert a single path (list of (x,y,layer)) into:
          - merged, cleaned segments (maximal straight runs)
          - list of turn points
        """
        if not path:
            return [], []

        pts = [(x, y) for (x, y, _) in path]
        # split into contiguous blocks on gaps
        blocks = []
        cur_block = [pts[0]]
        for i in range(1, len(pts)):
            if Geom.is_adjacent(pts[i - 1], pts[i]):
                cur_block.append(pts[i])
            else:
                blocks.append(cur_block)
                cur_block = [pts[i]]
        blocks.append(cur_block)

        merged_segments = []
        turns = []

        for block in blocks:
            if len(block) < 2:
                continue
            # small adjacent segments
            small_segs = [(block[j], block[j + 1]) for j in range(len(block) - 1)]
            # merge collinear consecutive small segments
            merged = Geom.merge_collinear_segments(small_segs)
            merged_segments.extend(merged)
            # detect turns inside the block
            for j in range(1, len(block) - 1):
                if Geom.is_turn(block[j - 1], block[j], block[j + 1]):
                    turns.append(block[j])

        if stop:
            breakpoint()
        return merged_segments, turns

    @staticmethod
    def extract_segments_from_all_paths(
        all_paths: List[List[Tuple[int, int, int]]],
        stop: bool = False,
    ) -> List[Tuple[List[Tuple[Tuple[int, int], Tuple[int, int]]], List[Tuple[int, int]]]]:
        """
        Process multiple paths: returns list of (merged_segments, turns) per path.
        """
        out = []
        for path in all_paths:
            segs, t = Geom.process_path(path, stop)
            out.append((segs, t))
        return out

