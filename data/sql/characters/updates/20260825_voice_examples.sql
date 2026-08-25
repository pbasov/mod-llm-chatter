-- Example dialogue lines per bot.
--
-- Persona conditioning in this module is adjectives end to end:
-- trait1/2/3 are randomly sampled words and `tone` is a 5-8 word
-- adjective phrase, both reaching the prompt as
--   Your personality: shrewd, adaptable, bawdy
-- with no example of the character actually speaking anywhere.
-- Trait lists are the weakest form of voice conditioning; a couple
-- of sample lines control voice far better. Generated once per bot,
-- alongside `tone`, and reused for the life of the character.
SET @db = DATABASE();

SET @s = (SELECT IF(
    (SELECT COUNT(*) FROM information_schema.COLUMNS
     WHERE TABLE_SCHEMA = @db
       AND TABLE_NAME = 'llm_bot_identities'
       AND COLUMN_NAME = 'voice_examples') = 0,
    'ALTER TABLE `llm_bot_identities` ADD COLUMN `voice_examples` TEXT DEFAULT NULL AFTER `tone`',
    'SELECT 1'));
PREPARE stmt FROM @s; EXECUTE stmt; DEALLOCATE PREPARE stmt;

SET @s = (SELECT IF(
    (SELECT COUNT(*) FROM information_schema.COLUMNS
     WHERE TABLE_SCHEMA = @db
       AND TABLE_NAME = 'llm_group_bot_traits'
       AND COLUMN_NAME = 'voice_examples') = 0,
    'ALTER TABLE `llm_group_bot_traits` ADD COLUMN `voice_examples` TEXT DEFAULT NULL AFTER `tone`',
    'SELECT 1'));
PREPARE stmt FROM @s; EXECUTE stmt; DEALLOCATE PREPARE stmt;
