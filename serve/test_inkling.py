"""Inkling 2: how the server closes its thought (the reasoning budget's wrap-up, reasoning_close_retry)."""
import unittest

from serve.frontend import THINK_END, InklingText
from serve.server import REASONING_CLOSE, REASONING_WRAP_UP, closed_with


class ThoughtClose(unittest.TestCase):
    def test_the_budget_ends_the_thought_with_inklings_own_token(self):
        # a </think> was text inside Inkling's thought: the parser took it as the end, the model thought on, and the
        # rest of the thought and its own end (a literal </think>) reached the answer
        wrap = closed_with(REASONING_WRAP_UP, THINK_END, inkling=True)
        self.assertNotIn(THINK_END, wrap)
        self.assertTrue(wrap.endswith("<|end_message|>"), wrap)
        ink = InklingText()
        out = ink.push("<|message_model|><|content_thinking|>Let me count.") + ink.push(wrap)
        self.assertEqual(out, "<think>Let me count.\n\nI have thought about this long enough; time to give my "
                              "answer." + THINK_END)
        # the answer is a message the model opens itself after it
        self.assertEqual(ink.push("<|message_model|><|content_text|>It is 3.<|end_message|>"), "It is 3.")

    def test_the_close_retry_too(self):
        self.assertEqual(closed_with(REASONING_CLOSE, THINK_END, inkling=True), "<|end_message|>")

    def test_the_other_models_keep_their_tags(self):
        self.assertEqual(closed_with(REASONING_CLOSE, THINK_END), REASONING_CLOSE)
        self.assertEqual(closed_with(REASONING_WRAP_UP, "<channel|>"),
                         REASONING_WRAP_UP.replace(THINK_END + "\n\n", "<channel|>"))


if __name__ == "__main__":
    unittest.main()
